# Backend isolation

Dataset is the interface for working with data in HypEx. Keep Spark and pandas
implementations isolated behind DatasetBackend and extensions.

- Direct operations on Spark or pandas objects belong in the corresponding
  DatasetBackend implementation (`hypex/dataset/backends/`) or in extensions
  (`hypex/extensions/`). This includes persistence, plan inspection, native
  dataframe accessors and conversion between native dataframe types.
- Experiment pipelines, splitters, executors and other shared code use the
  Dataset interface. Add or extend a backend operation when that interface is
  insufficient; avoid reaching through `raw_data` to a native dataframe API.
- Direct backend access elsewhere is reserved for exceptional, unique cases.
  Document the concrete reason and why a DatasetBackend operation or extension
  cannot serve that case. Existing exceptions are not a precedent for new ones.
- Backend tests may inspect native state to verify behavior that Dataset alone
  cannot expose, such as actual Spark cache ownership and storage levels.
