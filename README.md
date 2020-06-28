# simplecalc
Simple formula parser and evaluator

## Requirements

Requires Python 3.7 (probably)

## Usage

Runs as an interactive REPL.

Displays the stringified AST, the beautified expression and either the result or the detailed error.

```
$ python3 calc.py
> 2+2
parse.<locals>.BinOpNode(pos=1, op=BinOperator(symbol='+', priority=0, perform=<built-in function add>, rtl=False), left=parse.<locals>.NumberNode(pos=0, val=2), right=parse.<locals>.NumberNode(pos=2, val=2))
(2 + 2)
4

> 2^2^2
parse.<locals>.BinOpNode(pos=1, op=BinOperator(symbol='^', priority=2, perform=<built-in function pow>, rtl=True), left=parse.<locals>.NumberNode(pos=0, val=2), right=parse.<locals>.BinOpNode(pos=3, op=BinOperator(symbol='^', priority=2, perform=<built-in function pow>, rtl=True), left=parse.<locals>.NumberNode(pos=2, val=2), right=parse.<locals>.NumberNode(pos=4, val=2)))
(2 ^ (2 ^ 2))
16

> 2+2+2
parse.<locals>.BinOpNode(pos=3, op=BinOperator(symbol='+', priority=0, perform=<built-in function add>, rtl=False), left=parse.<locals>.BinOpNode(pos=1, op=BinOperator(symbol='+', priority=0, perform=<built-in function add>, rtl=False), left=parse.<locals>.NumberNode(pos=0, val=2), right=parse.<locals>.NumberNode(pos=2, val=2)), right=parse.<locals>.NumberNode(pos=4, val=2))
((2 + 2) + 2)
6

> 2+2+
      ↑
expected token, got EOL at 4
```

## Supported operators / functions

Unary:
- `-` : negation
- `~` : binary complement

Binary:
- `+` : sum
- `-` : difference
- `*` : product
- `/` : quotient
- `^` : exponent (right-to-left associative)

Functions:
- everything from `cmath`
- everything from `math` that isn't in `cmath` (`cos` is `cmath.cos` and not `math.cos`)

Constants:
- everything from `cmath` or `math`
- `i` : imaginary unit
