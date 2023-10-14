class MultiplePoolingMethodsException(Exception):
    """
    Is raised when more than one pooling method is
    set to True.

    """

    def __init__(self) -> None:
        self.message = f"""
        You can only set one pooling method to be true.
        Please check the arguments you have passed and 
        ensure you're setting unused pooling methods to
        False.
        """
        super().__init__(self.message)
