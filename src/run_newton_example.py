"""
This example is just to show how modules need to be run now, as oppposed to just executing them like
python3 newton_draft.py in their directory.
"""
from cs107_salad.Optimization.newton_draft import *

f = '2*x**3  - 2*x - 5'
newton(f, 2, 1.0e-8, 100)

# or can do 
main()