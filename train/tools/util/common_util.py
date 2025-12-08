import time
import multiprocessing

def run_with_retry(
        target_func,
        *args,
        max_retries: int = 10,
        delay_between_retries: int = 5,
        **kwargs
) -> bool:
    """
    Executes the specified function in a separate child process and retries on abnormal termination.

    Args:
        target_func: The function to be executed in the child process.
                     This function must be pickle-serializable.
        *args: Positional arguments to pass to target_func.
               These arguments must also be pickle-serializable.
        max_retries: Maximum number of retries for the function (default: 3).
        delay_between_retries: Time in seconds to wait before retrying (default: 5).
        **kwargs: Keyword arguments to pass to target_func.
                  These arguments must also be pickle-serializable.

    Returns:
        True if successfully executed and terminated with exit code 0 within max_retries,
        False otherwise.
    """
    attempt_count = 0

    while attempt_count <= max_retries:
        print(f"Attempt {attempt_count + 1}/{max_retries + 1}: Executing function '{target_func.__name__}'...")

        # Create a new process object
        # Using 'spawn' or 'forkserver' start methods instead of 'fork'
        # is more robust and stable, especially on Windows.
        # Refer to Python multiprocessing documentation for details.
        # We can safely set the start method to 'spawn' here.

        # In practice, it's recommended to set the start method once at the beginning of your script,
        # for example: multiprocessing.set_start_method('spawn', force=True)
        # For this example code, we directly use Process imported from multiprocessing.

        process = multiprocessing.Process(target=target_func, args=args, kwargs=kwargs)

        try:
            # Start the process
            process.start()

            # Wait for the process to terminate
            process.join()

            # Check the exit code
            if process.exitcode == 0:
                print(f" Success: Function '{target_func.__name__}' terminated successfully.")
                return True  # Terminated successfully
            else:
                print(
                    f" Failure: Function '{target_func.__name__}' terminated abnormally. Exit code: {process.exitcode}")
                # An abnormal exit code can result from sys.exit() or an unhandled exception in the child process.

        except Exception as e:
            print(f" Warning: An exception occurred during execution of process '{target_func.__name__}': {e}")
        finally:
            # Terminate the process if it's still alive (resource cleanup)
            if process.is_alive():
                print(f" Process is still running. Terminating it forcefully.")
                process.terminate()
                process.join()  # Wait for cleanup after forced termination

        attempt_count += 1
        if attempt_count <= max_retries:
            print(f"Retrying. Executing again in {delay_between_retries} seconds...")
            time.sleep(delay_between_retries)
        else:
            print(
                f"Maximum retry attempts ({max_retries}) reached. Not retrying function '{target_func.__name__}' anymore.")
            return False  # Exceeded maximum retry attempts

    return False  # This line should theoretically not be reached, but added for safety.
