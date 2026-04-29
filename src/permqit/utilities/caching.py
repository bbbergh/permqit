import abc
import functools
import inspect
import weakref
from typing import Callable, TypeVar, Any, Mapping

__all__ = ["cache", "cache_noargs", "WeakRefMemoize"]

T = TypeVar('T')
C = TypeVar('C')


def instance_cache_decorator(f: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to cache function results based on instance and parameters.
    The decorated function must have 'self' as its first argument.
    This is similar to functools.lru_cache, but stores the cache within the instance, so it gets cleared when the instance is deleted.
    :param f: Function to be decorated.
    :return: Decorated function with caching.
    """
    cache_attr = f"_cache_{f.__name__}"  # ty:ignore[unresolved-attribute]
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)

        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = f(self, *args, **kwargs)
        return cache[key]

    return wrapper

def instance_cache_no_argument_method_decorator(f: Callable[[C], T]) -> Callable[[C], T]:
    """
    A version of instance_cache for functions with no arguments (except for self).
    Similar to functools.cached_property but is not a property and so can be called as a method.
    :param f:
    :return:
    """
    cache_attr = f"_cache_{f.__name__}"  # ty:ignore[unresolved-attribute]
    @functools.wraps(f)
    def wrapper(self):
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, f(self))
        return getattr(self, cache_attr)
    return wrapper

cache_noargs = instance_cache_no_argument_method_decorator
cache = instance_cache_decorator




class WeakRefMemoize(abc.ABCMeta):
    """
    This is a metaclass (i.e. a subclass of type) used for memoizing objects by holding weakrefs.
    Whenever any instance of this type is constructed, the type and the parameters (args and kwargs passed to the constructor) are used as a cache key (if they are hashable) for this instance.
    Then, as long as this instance is still referenced somewhere else, whenever another instance with the same parameters is about to be constructed,
    it will not actually construct a new instance but reuse and return the previously constructed instance.

    Using weakrefs means that we do not keep the instance around forever, if there are no strong references in other code, it will be destroyed.
    If the paramters passed to the constructor are not hashable nothing is memoized and a new instance is always constructed.
    """
    cache = weakref.WeakValueDictionary()
    sentinel = object()

    def __intercept_new__(cls, *args, **kwargs):
        """This allows classes to inspect their __init__ arguments and return an already present object if so desired.
        If this returns something other than None, object creation will be aborted, and the return value of this function will be returned as the new object."""
        return None

    def __call__(cls, *args, **kwargs):
        if (obj := cls.__intercept_new__(*args, **kwargs)) is not None:
            return obj

        # We need to bind the args and kwargs to the signature, otherwise the signature will be different when arguments are supplied
        # via positional arguments or keyword arguments
        bound = inspect.signature(cls.__init__).bind('self', *args, **kwargs)
        bound.apply_defaults()
        del bound.arguments['self']
        cache_kwargs = cls.__process_init_args_for_cache_key__(**bound.arguments)
        try:
            key = cls, frozenset(cache_kwargs.items())
            _hash = hash(key)
        except TypeError:
            print(f"Cannot WeakRefMemoize {(cls, cache_kwargs.items())}: not hashable.")
            return super().__call__(*args, **kwargs)
        value = cls.cache.get(_hash, cls.sentinel) # At this point we have a strong reference and won't GC
        if value is not cls.sentinel:
            # print(f"Reusing cached {value}")
            return value
        value = super().__call__(*args, **kwargs)
        cls.cache[_hash] = value
        return value

    def __process_init_args_for_cache_key__(cls, **kwargs) -> Mapping[str, Any]:
        return kwargs

    @classmethod
    def clear_cache(cls):
        cls.cache.clear()

    @classmethod
    def clear_class(cls, target_cls):
        for k, v in cls.cache.items():
            if isinstance(v, target_cls):
                del cls.cache[k]