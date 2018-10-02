
def parse_check_ouput(out,prefix=None):
    result = {}
    for l in out.split('\n'):
        l = l.strip()
        p = l.split(':')
        if len(p)>1:
            if p[0].startswith('Obtained '):
                name = p[0][len('Obtained '):].strip()
                value = p[1].strip()
                l = value.split(' ')
                if len(l)>1:
                    value = l[0]
                if prefix is not None:
                    name = prefix+name
                result[name] = value
    return result
