import pathlib

p = pathlib.Path(__file__).parent

for file in p.glob('*demo*.py'):
    with open(file) as f:
        exec(f.read())
        print('{} ran successfully'.format(f))