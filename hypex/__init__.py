"""Tools to configure resources matcher."""
import warnings
import sys

from .ab_test.ab_tester import ABTest
from .ab_test.aa_tester import AATest
from .matcher import Matcher
from .__version__ import __version__

__all__ = ["Matcher", "AATest", "ABTest"]

warning_message = """
⚠️ ВНИМАНИЕ! В ближайшем обновлении HypEx перейдет на новую архитектуру! ⚠️
Это изменит интерфейс библиотеки, включая импорт, вызовы классов и методов.

🔗 Ознакомьтесь с актуальными туториалами: https://github.com/sb-ai-lab/HypEx/tree/dev/architecture_v11/examples/tutorials
🚀 Вы уже можете протестировать новую версию, установив её вручную:
    pip install --upgrade --pre hypex
🛑 Если не хотите обновляться, можете продолжить пользоваться старой версией:
    pip install hypex==0.1.10
Но она больше не будет поддерживаться.

⚠️ WARNING! HypEx will soon transition to a new architecture! ⚠️
This will change the library interface, including imports, class, and method calls.

🔗 Check out the latest tutorials: https://github.com/sb-ai-lab/HypEx/tree/dev/architecture_v11/examples/tutorials
🚀 You can already test the new version by installing it manually:
    pip install --upgrade --pre hypex
🛑 If you don’t want to update, you can continue using the old version:
    pip install hypex==0.1.10
However, it will no longer be supported.
"""

print(warning_message, file=sys.stderr)

warnings.warn(warning_message, FutureWarning, stacklevel=2)