def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.
    
    # Return the new value for x
    
	# We multiply gradient_of_x (the uphill direction) by learning_rate 
	# (the force of the push) and then subtract that from x to make the 
	# push go downhill.

    x= x- learning_rate*gradx
    
    
    
    return x