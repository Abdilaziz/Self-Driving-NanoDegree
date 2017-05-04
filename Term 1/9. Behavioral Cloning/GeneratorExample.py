def fibonacci():
    list = []
    while 1:
        if(len(list) < 2):
            list.append(1)
        else:
            list.append(list[-1]+list[-2])
        yield list # change this line so it yields its list instead of 1

our_generator = fibonacci()
my_output = []

for i in range(10):
    my_output = (next(our_generator))
    
print(my_output)