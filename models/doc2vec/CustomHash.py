import hashlib

def customHash(astring):
    """
    This is a rudimentary hash function designed to return a long integer corresponding to ASCII numbers for the string being hashed.
    Use this as input to doc2vec model to retain reproducibility.
    :param astring: Any string.
    :return: An integer
    """
    myStr=''
    asciiStr=myStr.join(str(ord(char)) for char in astring)
    asciiInt=int(asciiStr)
    # print('hash:'+astring+':',asciiInt)
    return asciiInt