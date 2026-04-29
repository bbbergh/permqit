import sys

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    # Resolve to a noop if Python < 3.13
    def deprecated(*args, **kwargs):
        def decorator(func):
            return func
        # Handle the case where it's used without arguments like @deprecated
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
