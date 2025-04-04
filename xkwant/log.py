from functools import wraps
import logging
import os

__all__ = ["log_function_call"]


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log the function call
        log_file_path = os.path.join(os.getcwd(), "xkwant.log")
        # Create a custom logger
        logger = logging.getLogger(func.__name__)
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a new file hand\ler each time the function is called
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        fh.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)
        logger.addHandler(console_handler)

        # Log all arguments
        for i, arg in enumerate(args):
            if hasattr(arg, "__dict__"):  # Check if the argument is a class instance
                logger.info(
                    f"Arg {i} is an instance of {arg.__class__.__name__} with properties {', '.join(f'{key} = {value}' for key, value in arg.__dict__.items() if key != 'H')}"
                )
            else:
                logger.info(f"Arg {i}: {arg}")

        # Log all keyword arguments
        for key, value in kwargs.items():
            if hasattr(value, "__dict__"):  # Check if the argument is a class instance
                logger.info(
                    f"Keyword arg {key} is an instance of {value.__class__.__name__} with properties {', '.join(f'{key} = {kvalue}' for key, kvalue in value.__dict__.items() if key != 'H')}"
                )
            else:
                logger.info(f"Keyword arg {key}: {value}")

        # Call the actual function
        result = func(*args, **kwargs)

        # Log the return value
        logger.info(f"Function {func.__name__} returned {result}")

        # Remove the handler after logging
        logger.removeHandler(fh)
        fh.close()

        return result

    return wrapper


# Example class
class ExampleClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Example function
@log_function_call
def example_function(a, b, instance, savepath="./scripts"):
    return a + b + instance.x + instance.y


# Example usage
if __name__ == "__main__":
    instance = ExampleClass(10, 20)
    result = example_function(1, 2, instance)
