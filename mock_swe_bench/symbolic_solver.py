# symbolic_solver.py

from sympy import symbols, Eq, solve, simplify, expand

def solve_equation(equation_str):
    x = symbols('x')
    try:
        lhs, rhs = equation_str.split('=')
        eq = Eq(lhs, rhs)
        result = solve(eq, x)
        return result
    except Exception as e:
        print("Error while solving equation:", e)
        return None

def simplify_expression(expr_str):
    x = symbols('x')
    try:
        expr = simplify(expr_str)
        return expr
    except Exception as e:
        print("Error while simplifying expression:", e)
        return None

def expand_expression(expr_str):
    x = symbols('x')
    try:
        expr = expand(expr_str)
        return expr
    except Exception as e:
        print("Error while expanding expression:", e)
        return None

def evaluate_expression(expr_str, value):
    x = symbols('x')
    try:
        expr = simplify(expr_str)
        result = expr.subs(x, value)
        return result
    except Exception as e:
        print("Error while evaluating expression:", e)
        return None
