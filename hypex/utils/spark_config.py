from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pyspark import SparkConf
from pyspark.sql import SparkSession

from ..config import MatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class SparkRecommendation:
    """A recommendation for configuring a Spark session."""

    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    priority: Literal["critical", "high", "medium", "low"] = "medium"


@dataclass
class SparkSettings:
    """Optimal Spark session settings for FAISS."""

    # Executor settings
    executor_instances: int = 10
    executor_cores: int = 4
    executor_memory: str = "4g"
    executor_memory_overhead: str = "1g"

    # Driver settings
    driver_memory: str = "4g"
    driver_cores: int = 2
    driver_max_result_size: str = "8g"

    # Partition settings
    shuffle_partitions: int = 100
    default_parallelism: int = 100
    max_partition_bytes: str = "128m"

    # FAISS-specific settings
    spark_files_max_bytes: str = "128m"
    broadcast_block_size: str = "4m"

    # Serialization
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    kryo_registration_required: bool = False

    # Memory management
    memory_fraction: float = 0.6
    memory_storage_fraction: float = 0.5

    # Additional settings
    extra_configs: dict[str, Any] = field(default_factory=dict)


class SparkSessionCalculator:
    """Calculator for optimal Spark session settings for FAISS.

    Usage example:
        >>> calculator = SparkSessionCalculator(
        ...     data_size_bytes=10_000_000_000,  # 10 GB
        ...     num_columns=8,
        ...     num_categorical_columns=2
        ... )
        >>> settings = calculator.calculate_optimal_settings()
        >>> calculator.print_recommendations()
        >>> calculator.apply_settings(spark_session)
    """

    # Calculation constants
    BYTES_PER_FLOAT = 4
    BYTES_PER_DECIMAL = 16  # decimal(38, 18) occupies ~16 bytes
    FAISS_INDEX_OVERHEAD = 1.5
    INDEX_FILE_SIZE_MULTIPLIER = 1.2

    # Redundancy and safety margins
    EXECUTOR_REDUNDANCY_FACTOR = 2
    MIN_EXECUTOR_MEMORY_GB = 4
    MIN_EXECUTOR_OVERHEAD_MB = 2048  # 2 ГБ минимум для native-кода
    MIN_DRIVER_MEMORY_GB = 8
    MIN_DRIVER_MAX_RESULT_SIZE_GB = 8

    # FAISS pipeline constants (must match hypex/config/config.py)
    # FAISS_SAMPLE_TARGET = 5_000_000
    # FAISS_CHUNK_SIZE = 4096

    # Mapping from SparkSettings parameters to SparkConf keys
    SETTINGS_TO_SPARK_CONF: ClassVar[dict[str, str]] = {
        "executor_instances": "spark.executor.instances",
        "executor_cores": "spark.executor.cores",
        "executor_memory": "spark.executor.memory",
        "executor_memory_overhead": "spark.executor.memoryOverhead",
        "driver_memory": "spark.driver.memory",
        "driver_cores": "spark.driver.cores",
        "driver_max_result_size": "spark.driver.maxResultSize",
        "shuffle_partitions": "spark.sql.shuffle.partitions",
        "default_parallelism": "spark.default.parallelism",
        "max_partition_bytes": "spark.sql.files.maxPartitionBytes",
        "spark_files_max_bytes": "spark.files.maxPartitionBytes",
        "broadcast_block_size": "spark.sql.broadcast.timeout",
        "serializer": "spark.serializer",
        "kryo_registration_required": "spark.kryo.registrationRequired",
        "memory_fraction": "spark.memory.fraction",
        "memory_storage_fraction": "spark.memory.storageFraction",
    }

    def __init__(
        self,
        data_size_bytes: int | None = None,
        num_rows: int | None = None,
        num_columns: int = 8,
        num_categorical_columns: int = 2,
        num_numeric_columns: int | None = None,
        target_executor_memory_gb: float = 4.0,
        target_executor_cores: int = 4,
    ):
        """Initialize the calculator.

        Args:
            data_size_bytes: Size of the data in bytes (if known)
            num_rows: Number of rows (if known)
            num_columns: Total number of columns
            num_categorical_columns: Number of categorical columns
            num_numeric_columns: Number of numeric columns (computed automatically)
            target_executor_memory_gb: Target executor memory in GB
            target_executor_cores: Target number of cores per executor
        """
        self.data_size_bytes = data_size_bytes
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_categorical_columns = num_categorical_columns
        self.num_numeric_columns = num_numeric_columns or (
            num_columns - num_categorical_columns
        )
        self.target_executor_memory_gb = target_executor_memory_gb
        self.target_executor_cores = target_executor_cores
        self._recommendations: list[SparkRecommendation] = []
        self._current_settings: dict[str, Any] = {}

    def _estimate_data_size(self) -> int:
        """Estimate the data size in bytes."""
        if self.data_size_bytes is not None:
            return self.data_size_bytes
        if self.num_rows is not None:
            numeric_size = self.num_numeric_columns * self.num_rows * 8
            categorical_size = self.num_categorical_columns * self.num_rows * 4
            return numeric_size + categorical_size
        return 1_000_000_000

    def _calculate_optimal_partitions(self, data_size_bytes: int) -> int:
        """Calculate the optimal number of partitions."""
        target_partition_size_mb = 150
        target_partition_size_bytes = target_partition_size_mb * 1024 * 1024
        num_partitions = max(1, data_size_bytes // target_partition_size_bytes)
        min_partitions = 10
        max_partitions = 500
        return max(min_partitions, min(num_partitions, max_partitions))

    def _calculate_optimal_executors(
        self, data_size_bytes: int, num_partitions: int
    ) -> int:
        """Calculate the optimal number of executors with redundancy.

        Uses EXECUTOR_REDUNDANCY_FACTOR (x2) to provide:
        - Fault tolerance for executor loss
        - Better load balancing under data skew
        - Headroom for long-running FAISS search operations
        """
        tasks_per_executor = self.target_executor_cores
        base_executors = max(1, num_partitions // tasks_per_executor)
        num_executors = base_executors * self.EXECUTOR_REDUNDANCY_FACTOR
        min_executors = 4   # increased for production stability
        max_executors = 50
        return max(min_executors, min(num_executors, max_executors))

    def _calculate_executor_memory(
        self, data_size_bytes: int, num_partitions: int
    ) -> str:
        """Calculate executor memory accounting for FAISS LRU cache behavior.

        KEY INSIGHT: In SparkFaissExtension._per_partition_predict(), each
        executor loads ALL partition indexes into a process-wide LRU cache
        (CachingIndex). Therefore, total memory must accommodate ALL indexes
        simultaneously, not just one partition's worth.

        Memory breakdown:
        1. all_indexes_memory: num_partitions × single_index_size
        2. working_data_memory: parallel tasks × partition_size
        3. chunk_buffers: parallel tasks × FAISS_CHUNK_SIZE × features × 4 bytes
        4. native_overhead: FAISS C++ + PyArrow + JVM metaspace (~1 GB baseline)
        """
        partition_size_bytes = data_size_bytes / num_partitions
        single_index_size_bytes = partition_size_bytes * self.FAISS_INDEX_OVERHEAD

        # CRITICAL: ALL indexes live in the executor's LRU cache simultaneously
        all_indexes_memory_bytes = single_index_size_bytes * num_partitions

        # Working memory for parallel tasks on this executor
        parallel_tasks = self.target_executor_cores
        working_data_bytes = partition_size_bytes * parallel_tasks

        # Query batch buffers (FAISS_CHUNK_SIZE rows × features × 4 bytes float32)
        chunk_buffer_bytes = (
            MatchingConfig.FAISS_CHUNK_SIZE * self.num_columns * self.BYTES_PER_FLOAT
            * parallel_tasks
        )

        # Native (off-heap) overhead for FAISS + PyArrow + JVM internals
        native_overhead_bytes = 1 * 1024 * 1024 * 1024  # 1 GB baseline

        total_memory_bytes = (
            all_indexes_memory_bytes
            + working_data_bytes
            + chunk_buffer_bytes
            + native_overhead_bytes
        )
        total_memory_gb = total_memory_bytes / (1024 ** 3)

        # Round up with safety margin, enforce minimum
        memory_gb = max(
            self.MIN_EXECUTOR_MEMORY_GB,
            int(total_memory_gb) + 2,   # +2 GB safety margin
        )
        return f"{memory_gb}g"

    def _calculate_memory_overhead(self, executor_memory_gb: int) -> str:
        """Calculate executor memory overhead for off-heap operations.

        For FAISS workloads, overhead must cover:
        - JVM metaspace and GC structures
        - PyArrow native buffers
        - Netty direct memory
        - SparkFiles temporary storage

        30% of executor memory with 2 GB minimum
        """
        overhead_mb = max(
            self.MIN_EXECUTOR_OVERHEAD_MB,
            int(executor_memory_gb * 1024 * 0.30),
        )
        return f"{overhead_mb}m"

    def _calculate_driver_memory(self, data_size_bytes: int) -> str:
        """Calculate driver memory based on dataset size.

        Driver memory consumers:
        1. FAISS IVF training sample (FAISS_SAMPLE_TARGET vectors × features × 4 bytes)
        2. Broadcast variables (index references, configs)
        3. Bug-triggered index collect() in ExperimentData.field_search()
           (the entire dataset's index is collected to driver)
        4. PySpark internal SQL catalyst + Arrow buffers
        """
        # 1. Sample for IVF training
        sample_bytes = (
            MatchingConfig.FAISS_SAMPLE_TARGET * self.num_columns * self.BYTES_PER_FLOAT
        )
        sample_gb = sample_bytes / (1024 ** 3)

        # 2. Index collect() bug impact: each row contributes 8 bytes (int64 index)
        if self.num_rows is not None:
            index_collect_bytes = self.num_rows * 8
        else:
            index_collect_bytes = data_size_bytes // self.num_columns
        index_collect_gb = index_collect_bytes / (1024 ** 3)

        # 3. Base overhead (catalyst, broadcast, Arrow)
        base_gb = 4

        total_gb = base_gb + sample_gb + index_collect_gb

        # Round up, enforce minimum
        driver_gb = max(self.MIN_DRIVER_MEMORY_GB, int(total_gb) + 2)
        return f"{driver_gb}g"

    def _calculate_driver_max_result_size(self) -> str:
        """Calculate spark.driver.maxResultSize.

        This limits the size of collect() results sent to the driver.
        Must be large enough to accommodate:
        - Index collect() from ExperimentData.field_search() bug
        - TTest/StatsTTest aggregation results
        - MatchingMetrics final results
        """
        if self.num_rows is not None:
            index_collect_bytes = self.num_rows * 8
            index_collect_gb = index_collect_bytes / (1024 ** 3)
        else:
            index_collect_gb = self._estimate_data_size() / (1024 ** 3)

        # At least 2x the expected collect size, minimum 8 GB
        result_size_gb = max(
            self.MIN_DRIVER_MAX_RESULT_SIZE_GB,
            int(index_collect_gb * 2) + 2,
        )
        return f"{result_size_gb}g"

    @staticmethod
    def _define_fs(session: SparkSession) -> str:
        """FIXED: removed erroneous `self` parameter from @staticmethod."""
        file_sys = None
        try:
            hadoop_conf = session.sparkContext._jsc.hadoopConfiguration()
            file_sys = hadoop_conf.get("fs.defaultFS", "file:///")
            if file_sys == "file:///":
                file_sys = None
        except Exception:
            pass
        if file_sys is None:
            try:
                file_sys = session.conf.get(
                    "spark.hadoop.fs.defaultFS",
                    session.conf.get("fs.defaultFS", "file:///"),
                )
            except Exception:
                file_sys = "file:///"
        # fsspec не поддерживает viewfs://. Мы заменяем его на hdfs://,
        # а резолв nameservice в реальные IP возьмет на себя libhdfs
        # благодаря добавлению hdfs-site.xml в CLASSPATH.
        if file_sys and file_sys.startswith("viewfs://"):
            file_sys = file_sys.replace("viewfs://", "hdfs://", 1)
        return file_sys or "file:///"

    def calculate_optimal_settings(self) -> SparkSettings:
        """Calculate optimal Spark session settings.

        Returns:
            SparkSettings with optimal parameters
        """
        data_size_bytes = self._estimate_data_size()
        num_partitions = self._calculate_optimal_partitions(data_size_bytes)
        num_executors = self._calculate_optimal_executors(
            data_size_bytes, num_partitions
        )
        executor_memory = self._calculate_executor_memory(
            data_size_bytes, num_partitions
        )
        executor_memory_gb = int(executor_memory.rstrip("g"))
        memory_overhead = self._calculate_memory_overhead(executor_memory_gb)

        # FIXED: driver memory is now calculated, not hardcoded
        driver_memory = self._calculate_driver_memory(data_size_bytes)
        driver_cores = max(4, self.target_executor_cores)
        driver_max_result_size = self._calculate_driver_max_result_size()

        settings = SparkSettings(
            executor_instances=num_executors,
            executor_cores=self.target_executor_cores,
            executor_memory=executor_memory,
            executor_memory_overhead=memory_overhead,
            driver_memory=driver_memory,
            driver_cores=driver_cores,
            driver_max_result_size=driver_max_result_size,
            shuffle_partitions=num_partitions,
            default_parallelism=num_partitions,
            max_partition_bytes="128m",
            spark_files_max_bytes="128m",
            broadcast_block_size="4m",
            serializer="org.apache.spark.serializer.KryoSerializer",
            kryo_registration_required=False,
            memory_fraction=0.6,
            memory_storage_fraction=0.5,
            extra_configs={
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.skewJoin.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.kryoserializer.buffer.max": "512m",
                # FIXED: maxResultSize now matches driver memory calculation
                "spark.driver.maxResultSize": driver_max_result_size,
            },
        )
        return settings

    def check_current_settings(
        self, spark_session: SparkSession
    ) -> dict[str, Any]:
        """Check current Spark session settings.

        Args:
            spark_session: Active Spark session

        Returns:
            Dictionary with current settings
        """
        if spark_session is None:
            return {}
        conf = spark_session.sparkContext.getConf()
        settings = {
            "executor_instances": int(conf.get("spark.executor.instances", "0")),
            "executor_cores": int(conf.get("spark.executor.cores", "0")),
            "executor_memory": conf.get("spark.executor.memory", "unknown"),
            "executor_memory_overhead": conf.get(
                "spark.executor.memoryOverhead", "unknown"
            ),
            "driver_memory": conf.get("spark.driver.memory", "unknown"),
            "driver_cores": int(conf.get("spark.driver.cores", "0")),
            # FIXED: added maxResultSize check
            "driver_max_result_size": conf.get(
                "spark.driver.maxResultSize", "1g"
            ),
            "shuffle_partitions": int(
                conf.get("spark.sql.shuffle.partitions", "200")
            ),
            "default_parallelism": int(
                conf.get("spark.default.parallelism", "0")
            ),
            "max_partition_bytes": conf.get(
                "spark.sql.files.maxPartitionBytes", "128m"
            ),
            "serializer": conf.get("spark.serializer", "unknown"),
        }
        self._current_settings = settings
        return settings

    def generate_recommendations(
        self,
        current_settings: dict[str, Any],
        optimal_settings: SparkSettings,
    ) -> list[SparkRecommendation]:
        """Generate configuration recommendations.

        Args:
            current_settings: Current settings
            optimal_settings: Optimal settings

        Returns:
            List of recommendations
        """
        recommendations = []

        if (
            current_settings.get("executor_instances", 0)
            < optimal_settings.executor_instances
        ):
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.executor.instances",
                    current_value=current_settings.get("executor_instances"),
                    recommended_value=optimal_settings.executor_instances,
                    reason=(
                        "Increase the number of executors for parallel "
                        "partition processing"
                    ),
                    priority="high",
                )
            )

        if (
            current_settings.get("executor_cores", 0)
            > optimal_settings.executor_cores
        ):
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.executor.cores",
                    current_value=current_settings.get("executor_cores"),
                    recommended_value=optimal_settings.executor_cores,
                    reason=(
                        "Reduce the number of cores per executor to prevent "
                        "OOM when loading indexes"
                    ),
                    priority="critical",
                )
            )

        current_memory = current_settings.get("executor_memory", "unknown")
        if current_memory != optimal_settings.executor_memory:
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.executor.memory",
                    current_value=current_memory,
                    recommended_value=optimal_settings.executor_memory,
                    reason="Optimize executor memory for loading FAISS indexes",
                    priority="high",
                )
            )

        # FIXED: added driver memory recommendation
        current_driver_memory = current_settings.get(
            "driver_memory", "unknown"
        )
        if current_driver_memory != optimal_settings.driver_memory:
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.driver.memory",
                    current_value=current_driver_memory,
                    recommended_value=optimal_settings.driver_memory,
                    reason=(
                        "Driver memory must accommodate FAISS training "
                        "sample, index collect(), and broadcast variables"
                    ),
                    priority="critical",
                )
            )

        # FIXED: added maxResultSize recommendation
        current_max_result = current_settings.get(
            "driver_max_result_size", "unknown"
        )
        if current_max_result != optimal_settings.driver_max_result_size:
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.driver.maxResultSize",
                    current_value=current_max_result,
                    recommended_value=optimal_settings.driver_max_result_size,
                    reason=(
                        "Prevents GC overhead limit exceeded during "
                        "collect() operations (TTest, field_search bug)"
                    ),
                    priority="critical",
                )
            )

        if (
            current_settings.get("shuffle_partitions", 200)
            != optimal_settings.shuffle_partitions
        ):
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.sql.shuffle.partitions",
                    current_value=current_settings.get("shuffle_partitions"),
                    recommended_value=optimal_settings.shuffle_partitions,
                    reason=(
                        "Optimize partition count for balance between "
                        "index size and parallelism"
                    ),
                    priority="high",
                )
            )

        if current_settings.get("serializer") != optimal_settings.serializer:
            recommendations.append(
                SparkRecommendation(
                    parameter="spark.serializer",
                    current_value=current_settings.get("serializer"),
                    recommended_value=optimal_settings.serializer,
                    reason="Use KryoSerializer for more efficient serialization",
                    priority="medium",
                )
            )

        self._recommendations = recommendations
        return recommendations

    def print_recommendations(self) -> None:
        """Print recommendations to the console."""
        if not self._recommendations:
            print("✓ All settings are optimal")
            return

        print("\n" + "=" * 80)
        print("SPARK SESSION CONFIGURATION RECOMMENDATIONS")
        print("=" * 80)
        for rec in self._recommendations:
            priority_marker = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(rec.priority, "⚪")
            print(f"\n{priority_marker} [{rec.priority.upper()}] {rec.parameter}")
            print(f"   Current value: {rec.current_value}")
            print(f"   Recommended value: {rec.recommended_value}")
            print(f"   Reason: {rec.reason}")
        print("\n" + "=" * 80)

    def apply_settings(
        self, spark_session: SparkSession, settings: SparkSettings
    ) -> None:
        """Apply settings to a Spark session.

        Args:
            spark_session: Active Spark session
            settings: Settings to apply
        """
        conf = spark_session.sparkContext.getConf()
        runtime_settings = {
            "spark.sql.shuffle.partitions": str(settings.shuffle_partitions),
            "spark.default.parallelism": str(settings.default_parallelism),
            "spark.sql.files.maxPartitionBytes": settings.max_partition_bytes,
        }
        for key, value in runtime_settings.items():
            conf.set(key, value)
        for key, value in settings.extra_configs.items():
            conf.set(key, str(value))
        print("✓ Runtime settings have been applied to the session")
        print("⚠ Executor settings can only be changed when creating the session")

    @staticmethod
    def create_optimal_session(settings: SparkSettings) -> SparkSession:
        """Create a new Spark session with optimal settings.

        Args:
            settings: Settings for the session

        Returns:
            New Spark session
        """
        if SparkSession is None:
            raise ImportError("PySpark is not installed")

        builder = (
            SparkSession.builder.appName("HypEx-FAISS")
            .config("spark.executor.instances", str(settings.executor_instances))
            .config("spark.executor.cores", str(settings.executor_cores))
            .config("spark.executor.memory", settings.executor_memory)
            .config(
                "spark.executor.memoryOverhead",
                settings.executor_memory_overhead,
            )
            .config("spark.driver.memory", settings.driver_memory)
            .config("spark.driver.cores", str(settings.driver_cores))
            .config(
                "spark.driver.maxResultSize",
                settings.driver_max_result_size,
            )
            .config(
                "spark.sql.shuffle.partitions",
                str(settings.shuffle_partitions),
            )
            .config(
                "spark.default.parallelism",
                str(settings.default_parallelism),
            )
            .config(
                "spark.sql.files.maxPartitionBytes",
                settings.max_partition_bytes,
            )
            .config("spark.serializer", settings.serializer)
            .config(
                "spark.kryo.registrationRequired",
                str(settings.kryo_registration_required).lower(),
            )
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
        )
        for key, value in settings.extra_configs.items():
            builder = builder.config(key, str(value))
        return builder.getOrCreate()

    def optimize_config(
        self, config: SparkConf | dict[str, str] | None = None
    ) -> SparkConf:
        """Optimize a Spark session configuration according to optimal settings.

        Takes an existing configuration, applies optimal settings for FAISS,
        keeps all other configurations unchanged, prints the new configuration,
        and logs all changes with explanations of the reasons.

        Args:
            config: Initial Spark session configuration. Can be:
                - SparkConf object
                - Dictionary {key: value}
                - None (an empty SparkConf is created)

        Returns:
            SparkConf: Optimized configuration for creating a new session

        Example:
            >>> calculator = SparkSessionCalculator(
            ...     data_size_bytes=10_000_000_000,
            ...     num_columns=8,
            ...     num_categorical_columns=2
            ... )
            >>> from pyspark import SparkConf
            >>> conf = SparkConf().setAppName("MyApp").setMaster("yarn")
            >>> optimized_conf = calculator.optimize_config(conf)
        """
        if SparkConf is None:
            raise ImportError("PySpark is not installed")

        # Get optimal settings
        optimal_settings = self.calculate_optimal_settings()

        # Convert the input configuration to a dictionary
        if config is None:
            current_config = {}
        elif isinstance(config, dict):
            current_config = dict(config)
        elif isinstance(config, SparkConf):
            current_config = {key: value for key, value in config.getAll()}
        else:
            raise TypeError(
                f"config must be SparkConf, dict, or None, "
                f"received: {type(config)}"
            )

        # Create a new configuration based on the original
        optimized_conf = SparkConf()

        # Copy all existing parameters
        for key, value in current_config.items():
            optimized_conf.set(key, value)

        # Apply optimal settings and log the changes
        changes_log = []

        # FIXED: added driver memory, driver cores, and maxResultSize rules
        optimization_rules = [
            {
                "key": "spark.executor.instances",
                "value": str(optimal_settings.executor_instances),
                "reason": (
                    f"Optimal number of executors for parallel processing of "
                    f"{self._calculate_optimal_partitions(self._estimate_data_size())} "
                    f"partitions with {self.EXECUTOR_REDUNDANCY_FACTOR}x "
                    f"redundancy factor. Provides balance between parallelism "
                    f"and fault tolerance."
                ),
            },
            {
                "key": "spark.executor.cores",
                "value": str(optimal_settings.executor_cores),
                "reason": (
                    "Limiting cores per executor to prevent OOM when "
                    "simultaneously loading multiple FAISS indexes into "
                    "memory. Recommended ≤4 cores."
                ),
            },
            {
                "key": "spark.executor.memory",
                "value": optimal_settings.executor_memory,
                "reason": (
                    f"Executor memory calculated for loading ALL FAISS "
                    f"indexes (~{self._estimate_data_size() / (1024**3):.1f} GB "
                    f"× {self.FAISS_INDEX_OVERHEAD} overhead) into the "
                    f"process-wide LRU cache simultaneously."
                ),
            },
            {
                "key": "spark.executor.memoryOverhead",
                "value": optimal_settings.executor_memory_overhead,
                "reason": (
                    "Memory overhead for JVM, FAISS native code, PyArrow, "
                    "and off-heap operations. 30% of executor memory, "
                    "minimum 2 GB."
                ),
            },
            {
                "key": "spark.driver.memory",
                "value": optimal_settings.driver_memory,
                "reason": (
                    "Driver memory for FAISS IVF training sample "
                    f"({MatchingConfig.FAISS_SAMPLE_TARGET} vectors), index collect() "
                    "from ExperimentData.field_search(), broadcast variables, "
                    "and PySpark catalyst internals."
                ),
            },
            {
                "key": "spark.driver.cores",
                "value": str(optimal_settings.driver_cores),
                "reason": (
                    "Driver cores must be at least equal to executor cores "
                    "for efficient task scheduling and broadcast coordination."
                ),
            },
            {
                "key": "spark.driver.maxResultSize",
                "value": optimal_settings.driver_max_result_size,
                "reason": (
                    "Prevents 'GC overhead limit exceeded' during collect() "
                    "operations. Must accommodate index collect() from "
                    "field_search() bug and TTest aggregation results."
                ),
            },
            {
                "key": "spark.sql.shuffle.partitions",
                "value": str(optimal_settings.shuffle_partitions),
                "reason": (
                    f"Number of partitions for shuffle operations. Optimized "
                    f"for data size "
                    f"{self._estimate_data_size() / (1024**3):.1f} GB."
                ),
            },
            {
                "key": "spark.default.parallelism",
                "value": str(optimal_settings.default_parallelism),
                "reason": (
                    "Base parallelism level for RDD operations. Should match "
                    "shuffle.partitions for consistency."
                ),
            },
            {
                "key": "spark.serializer",
                "value": optimal_settings.serializer,
                "reason": (
                    "KryoSerializer provides more efficient serialization "
                    "compared to JavaSerializer (2-10x faster, smaller "
                    "size). Critical for FAISS."
                ),
            },
            {
                "key": "spark.memory.fraction",
                "value": str(optimal_settings.memory_fraction),
                "reason": (
                    "Memory fraction for execution and storage. 60% provides "
                    "balance between data caching and FAISS operation "
                    "execution."
                ),
            },
            {
                "key": "spark.memory.storageFraction",
                "value": str(optimal_settings.memory_storage_fraction),
                "reason": (
                    "Memory fraction for storage (RDD caching). 50% allows "
                    "storing FAISS indexes in memory without evicting "
                    "execution memory."
                ),
            },
        ]

        # Apply each rule and log the changes
        for rule in optimization_rules:
            key = rule["key"]
            new_value = rule["value"]
            reason = rule["reason"]
            old_value = current_config.get(key)

            if old_value is None:
                optimized_conf.set(key, new_value)
                changes_log.append(
                    {
                        "action": "ADDED",
                        "parameter": key,
                        "old_value": None,
                        "new_value": new_value,
                        "reason": reason,
                    }
                )
            elif str(old_value) != str(new_value):
                optimized_conf.set(key, new_value)
                changes_log.append(
                    {
                        "action": "CHANGED",
                        "parameter": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reason": reason,
                    }
                )
            else:
                pass

        # Apply additional settings from extra_configs
        for key, value in optimal_settings.extra_configs.items():
            old_value = current_config.get(key)
            if old_value is None:
                optimized_conf.set(key, str(value))
                changes_log.append(
                    {
                        "action": "ADDED",
                        "parameter": key,
                        "old_value": None,
                        "new_value": str(value),
                        "reason": "Additional setting for FAISS pipeline",
                    }
                )
            elif str(old_value) != str(value):
                optimized_conf.set(key, str(value))
                changes_log.append(
                    {
                        "action": "CHANGED",
                        "parameter": key,
                        "old_value": old_value,
                        "new_value": str(value),
                        "reason": "Optimization of additional setting for FAISS",
                    }
                )

        # Print the change log to the console
        self._print_optimization_log(changes_log)
        return optimized_conf

    def _print_optimization_log(self, changes_log: list[dict]) -> None:
        """Print the optimization log to the console.

        Args:
            changes_log: List of changes with justifications
        """
        if not changes_log:
            print("\n" + "=" * 80)
            print("✓ CONFIGURATION IS ALREADY OPTIMAL — NO CHANGES REQUIRED")
            print("=" * 80)
            return

        print("\n" + "=" * 80)
        print("SPARK SESSION CONFIGURATION OPTIMIZATION LOG")
        print("=" * 80)

        added_count = sum(1 for c in changes_log if c["action"] == "ADDED")
        changed_count = sum(1 for c in changes_log if c["action"] == "CHANGED")

        print(f"\n📊 TOTAL CHANGES: {len(changes_log)}")
        print(f"   ➕ Added: {added_count}")
        print(f"   🔄 Changed: {changed_count}")
        print("\n" + "-" * 80)

        for i, change in enumerate(changes_log, 1):
            action_icon = "➕" if change["action"] == "ADDED" else "🔄"
            action_text = "ADDED" if change["action"] == "ADDED" else "CHANGED"
            print(f"\n{action_icon} [{action_text}] {i}. {change['parameter']}")
            if change["action"] == "CHANGED":
                print(f"   Was: {change['old_value']}")
            print(f"   Now: {change['new_value']}")
            print(f"   Reason: {change['reason']}")

        print("\n" + "=" * 80)
        print("✓ CONFIGURATION HAS BEEN OPTIMIZED FOR THE FAISS PIPELINE")
        print("=" * 80 + "\n")
