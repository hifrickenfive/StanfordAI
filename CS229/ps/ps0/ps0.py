def arg_one(*args):
    for stuff in args:
        print(stuff)

def arg_two(one, two, *args):
    print(one)
    print(two)
    print(args)

def kwargs(v, **kwargs):
    if 'power' in kwargs:
        return v ** kwargs['power']
    else:
        return v

if __name__ == '__main__':

    my_list = ['honda', 'maserati', 'ferrari']
    arg_one(*my_list) # Need * if calling from list
    print('----')
    arg_one('required1', 'maserati', 'ferrari') # maserati and ferarri are in args
    print('----')
    arg_two('required1', 'required2', 'ferrari',) # ferrari is the only item in args
    print('----')
    print(kwargs(10, key1='extra 1', key2='extra 2', power=2))
    print('----')
    print(kwargs(10, key1=5))