# Decorators in Python

A **decorator** is a callable that takes a function and returns a new
function, usually one that wraps the original with extra behaviour. Decorators
are built on two ideas: functions are objects, and closures.

## Functions are first-class objects

In Python, functions are ordinary objects. You can assign them to names,
pass them as arguments, return them from other functions, and store them in
data structures.

```python
def shout(text):
    return text.upper()

yell = shout                # assign to another name
yell("hi")                  # 'HI'

def apply(func, value):     # pass as an argument
    return func(value)

apply(shout, "hey")         # 'HEY'

handlers = {"loud": shout, "quiet": str.lower}
handlers["loud"]("ok")      # 'OK'
```

## A basic decorator

A decorator defines an inner `wrapper` function that closes over `func`, then
returns `wrapper`:

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper
```

The `@` syntax is shorthand for reassigning the name:

```python
@log_calls
def add(a, b):
    return a + b

# exactly equivalent to:
# add = log_calls(add)

add(2, 3)
# Calling add with (2, 3) {}
# add returned 5
```

Use `*args, **kwargs` so the wrapper accepts any signature, and always
**return the result**. Forgetting `return result` makes the decorated
function silently return `None`.

## Preserving metadata with `functools.wraps`

The wrapper replaces the original function, so its metadata is lost:

```python
add.__name__   # 'wrapper'
add.__doc__    # None
```

This breaks debugging, logging, and documentation tools. `functools.wraps`
copies `__name__`, `__doc__`, `__module__`, `__qualname__` and more from the
original onto the wrapper, and sets `__wrapped__`:

```python
import functools

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def add(a, b):
    """Add two numbers."""
    return a + b

add.__name__      # 'add'
add.__doc__       # 'Add two numbers.'
add.__wrapped__   # the original, undecorated add
```

Rule of thumb: every decorator you write should use `@functools.wraps`.

## Decorators with arguments

To pass arguments to a decorator, add one more layer: a **decorator factory**
that takes the arguments and returns the actual decorator.

```python
import functools

def repeat(times):                 # factory: takes the arguments
    def decorator(func):           # the real decorator
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say(msg):
    print(msg)

say("hi")    # prints "hi" three times

# equivalent to: say = repeat(times=3)(say)
```

Note the parentheses: `@repeat(3)` calls the factory first. Writing `@repeat`
without arguments would pass the function in as `times`, which is a common
bug.

## Stacking decorators

Decorators apply bottom-up (the one closest to `def` wraps first):

```python
@log_calls
@repeat(2)
def ping():
    print("ping")

# ping = log_calls(repeat(2)(ping))
```

## Key takeaways

- Functions are objects, so they can be passed in and returned.
- A decorator takes a function and returns a wrapper (a closure over `func`).
- `@deco` is shorthand for `f = deco(f)`.
- Always use `functools.wraps` to keep the original's name and docstring.
- Decorators with arguments need three levels: factory → decorator → wrapper.
