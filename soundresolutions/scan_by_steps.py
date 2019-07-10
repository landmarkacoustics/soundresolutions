# Copyright (C) 2018 by Landmark Acoustics LLC

def try_to_read(source, dest):
    r'''Writes the contents of `source` into `dest`

        Tries to put the entirety of `source` into `dest`. The write is
    left-justified in dest, and if `len(source) > len(dest)` then it
    puts the last `len(dest)` entries of `source` into `dest`.

    Parameters
    ----------
    source : array-like
        This should be indexable
    dest : array-like
        This should be indexable and writable

    Returns
    -------
    n_to_read : int
        The actual amount of data moved from `source` to `dest`.

    Examples
    --------
    >>> foo = np.zeros(5)
    >>> bar = np.linspace(1,3,3)
    >>> try_to_read(bar, foo)
    3
    >>> foo
    array([ 1.,  2.,  3.,  0.,  0.])

    '''
    
    dest_length = len(dest)
    n_to_read = len(source)
    if n_to_read <= dest_length:
        dest[:n_to_read] = source[:n_to_read]
    else:
        dest[:] = source[:n_to_read][-dest_length:]
    
    return n_to_read

def scan_by_steps(source, dest, n_steps, pad = 0):
    r'''Yields several extracts from `source`, each the size of `dest`.
    
    Parameters
    ----------
    source : array-like
        Should be indexable. Its length determines the number of extracts.
    dest : array-like
        Should be indexable and writable. Each extract has its length.
    n_steps : int
        The number of data units between the starts of each extract
    pad : any, optional
        The value to pad the beginning and (maybe) ending extracts.
    if n_steps < 1:
        raise ValueError('the number of steps must be positive')
    
    Raises
    ------
    ValueError : if n_steps is below 1

    Yields
    ------
    dest : the current contents of the destination buffer

    Examples
    --------
    >>> foo = np.linspace(-10,-1,10)
    >>> bar = np.zeros(5)
    >>> np.array([x.copy() for x in scan_by_steps(foo,bar,2)])
    array([[  0.,   0., -10.,  -9.,  -8.],
           [-10.,  -9.,  -8.,  -7.,  -6.],
           [ -8.,  -7.,  -6.,  -5.,  -4.],
           [ -6.,  -5.,  -4.,  -3.,  -2.],
           [ -4.,  -3.,  -2.,  -1.,   0.]])

    '''
    
    dest_length = len(dest)
    offset = max(dest_length - n_steps, 0)
    half = dest_length // 2
    dest[:half] = pad
    n_in = try_to_read(source[:dest_length - half], dest[half:])
    dest_content_size = half + n_in
    source = source[n_in:]

    while dest_content_size == dest_length:
        yield dest
        if offset:
            dest[:offset] = dest[n_steps:]
        
        n_in = try_to_read(source[:n_steps],dest[offset:])
        dest_content_size += n_in - n_steps
        source = source[n_in:]
    
    while dest_content_size > half:
        if offset == 0:
            dest[:dest_content_size] = dest[-dest_content_size:]
        dest[dest_content_size:] = pad
        yield dest
        if offset:
            dest[:offset] = dest[n_steps:]
        dest_content_size -= n_steps
