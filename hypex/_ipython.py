from IPython.core.getipython import get_ipython

def _jupyter_hypex_completer(ipython, event):
    """Кастомный комплитер для методов Hypex"""
    from hypex import Hypex
    if isinstance(event.obj, Hypex):
        return dir(Hypex)
    return []

def load_ipython_extension():
    """Регистрируем комплитер в IPython"""
    ip = get_ipython()
    if ip:
        ip.set_hook('complete_command', _jupyter_hypex_completer, re_key=r'.*')