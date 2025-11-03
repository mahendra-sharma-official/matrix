# C++ matrix class with multithreading support 

# Features
1. Multithreading can be used for matrix mathematical operations (needs to be enabled explicitly).
2. Common matrix mathematical operations.
3. Common matrix functions and helpers.
4. Runtime checks for error (needs to be enabled explicitly).
5. Header only so easy to setup

# Notes
	MULTITHREADING_THRESHOLD // set its value as the number of elements needed to use multithreading otherwise uses single thread
	use_multithreading_ // needs to be enabled to use multithreading otherwise even if threshold is reached multithreading won't be used

# Extra
Made this so that it will be easier to do the main project instead of writing matrix class again. 