import pathlib

p = pathlib.Path('run_all.py').parent

for file in p.glob('*demo*.py'):
    with open(file) as f:
        exec(f.read())
        print('{} ran successfully'.format(f))