from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum
import operator
import string


@dataclass
class BinOperator:
    symbol: str
    priority: int
    perform: Callable
    rtl: bool = False


operators = [
    BinOperator("+", 0, operator.add),
    BinOperator("-", 0, operator.sub),
    BinOperator("*", 1, operator.mul),
    BinOperator("/", 1, operator.truediv),
    BinOperator("^", 2, operator.pow, True)
]

max_priority = max(op.priority for op in operators)


def ops_by_priority(priority):
    return [op for op in operators if op.priority == priority]


@dataclass
class UnaryOperator:
    symbol: str
    perform: Callable


unary_operators = [
    UnaryOperator("-", operator.neg),
    UnaryOperator("~", operator.inv)
]

ops_syms = [op.symbol for op in operators + unary_operators]


class TokenType(Enum):
    NUMBER = 1
    IDENTIFIER = 2
    PARENTHESIS = 3
    OPERATION = 4
    EOL = 5


@dataclass
class Token:
    type: TokenType
    val: Any
    pos: int

    def __str__(self):
        return f"({self.type.name}, {repr(self.val)})"


class ParseError(Exception):
    pos: int

    def __init__(self, msg, pos=None):
        super().__init__(msg)
        self.pos = pos


class UndeclaredVarError(ParseError):
    var: str

    def __init__(self, var, pos=None):
        super().__init__(f"undeclared variable '{var}'", pos)
        self.var = var


class UndeclaredFuncError(ParseError):
    def __init__(self, var, pos=None):
        super().__init__(f"undeclared function '{var}'", pos)


def stringify_rule(type: TokenType, val: Any = None):
    if val is None:
        return type.name
    else:
        return f"'{val}'"


def tokenize(inp: str):
    tokens = []

    index = 0

    def skip_spaces():
        nonlocal index
        while inp[index].isspace():
            index += 1

    def has():
        return index < len(inp)

    def peek():
        return inp[index]

    def read():
        nonlocal index
        index += 1
        return inp[index - 1]

    valid_number_chars = "0123456789."

    def read_number():
        res = ""
        pos = index

        while True:
            res += read()
            if not has() or peek() not in valid_number_chars:
                break

        return Token(TokenType.NUMBER, float(res) if "." in res else int(res), pos)

    def read_identifier():
        res = ""
        pos = index

        while True:
            res += read()
            if not has() or peek() not in string.ascii_letters + string.digits:
                break

        return Token(TokenType.IDENTIFIER, res, pos)

    while index < len(inp):
        skip_spaces()

        next = peek()
        pos = index

        if next in ops_syms:
            tok = Token(TokenType.OPERATION, read(), pos)
        elif next in "()":
            tok = Token(TokenType.PARENTHESIS, read(), pos)
        elif next in valid_number_chars:
            tok = read_number()
        elif next in string.ascii_letters:
            tok = read_identifier()
        else:
            raise ParseError(f"invalid character '{next}'", index)

        tokens.append(tok)

    tokens.append(Token(TokenType.EOL, None, index))

    return tokens


def parse(tokens):
    @dataclass
    class Node:
        pos: int

        def write(self):
            raise NotImplementedError

        def eval(self, ctx):
            try:
                return self._eval(ctx)
            except Exception as e:
                if not isinstance(e, ParseError):
                    raise ParseError(str(e), self.pos)
                else:
                    raise

        def _eval(self, ctx):
            raise NotImplementedError

    @dataclass
    class BinOpNode(Node):
        op: BinOperator
        left: Node
        right: Node

        def write(self):
            return f"({self.left.write()} {self.op.symbol} {self.right.write()})"

        def _eval(self, ctx):
            return self.op.perform(self.left.eval(ctx), self.right.eval(ctx))

    @dataclass
    class UnaryOpNode(Node):
        op: UnaryOperator
        expr: Node

        def write(self):
            return f"({self.op.symbol}{self.expr.write()})"

        def _eval(self, ctx):
            return self.op.perform(self.expr.eval(ctx))

    @dataclass
    class CallNode(Node):
        head: str
        arg: Node

        def write(self):
            return f"({self.head}({self.arg.write()}))"

        def _eval(self, ctx):
            try:
                return ctx.get_func(self.head)(self.arg.eval(ctx))
            except KeyError:
                try:
                    return BinOpNode(self.pos, operators[2], IdentifierNode(self.pos, self.head), self.arg).eval(ctx)
                except UndeclaredVarError:
                    raise UndeclaredFuncError(self.head, self.pos)

    @dataclass
    class NumberNode(Node):
        val: float

        def write(self):
            return str(self.val)

        def _eval(self, ctx):
            return self.val

    @dataclass
    class IdentifierNode(Node):
        val: str

        def write(self):
            return self.val

        def _eval(self, ctx):
            try:
                return ctx.get_var(self.val)
            except KeyError:
                raise UndeclaredVarError(self.val, self.pos)

    index = 0

    def has():
        return index < len(tokens) - 1

    def current():
        if not has():
            raise ParseError("expected token, got EOL", tokens[index].pos)
        return tokens[index]

    def match(type: TokenType, val: Any = None):
        return has() and tokens[index].type == type and (val is None or tokens[index].val == val)

    def accept(type: TokenType, val: Any = None):
        nonlocal index
        if match(type, val):
            index += 1
            return True
        return False

    def expect(type: TokenType, val: Any = None):
        nonlocal index
        if match(type, val):
            index += 1
            return tokens[index - 1]
        if not has():
            raise ParseError(
                f"expected {stringify_rule(type, val)}, got EOL", tokens[index].pos)
        else:
            raise ParseError(
                f"expected {stringify_rule(type, val)}, got {stringify_rule(current().type, current().val)}")

    def parse_bin(priority=0):
        if priority > max_priority:
            return parse_unary()

        left = parse_bin(priority + 1)
        ops = ops_by_priority(priority)

        while has() and current().type == TokenType.OPERATION:
            pos = current().pos
            for op in ops:
                if accept(TokenType.OPERATION, op.symbol):
                    right = parse_bin(priority + 1 - op.rtl)
                    left = BinOpNode(pos, op, left, right)
                    break
            else:
                break

        return left

    def parse_unary():
        for op in unary_operators:
            pos = index
            if accept(TokenType.OPERATION, op.symbol):
                return UnaryOpNode(pos, op, parse_unary())

        return parse_call()

    def parse_call():
        head = parse_term()

        if isinstance(head, IdentifierNode) and accept(TokenType.PARENTHESIS, "("):
            arg = parse_expr()
            expect(TokenType.PARENTHESIS, ")")
            return CallNode(head.pos, head.val, arg)

        return head

    def parse_term():
        token = current()

        if token.type == TokenType.NUMBER:
            return NumberNode(token.pos, expect(TokenType.NUMBER).val)
        elif token.type == TokenType.IDENTIFIER:
            return IdentifierNode(token.pos, expect(TokenType.IDENTIFIER).val)
        elif accept(TokenType.PARENTHESIS, "("):
            val = parse_expr()
            expect(TokenType.PARENTHESIS, ")")
            return val
        else:
            raise ParseError(
                f"expected term, got {stringify_rule(token.type, token.val)}", token.pos)

    def parse_expr():
        nonlocal index
        left = parse_bin()
        if index < len(tokens):
            old = index
            try:
                right = parse_expr()
                return BinOpNode(right.pos, operators[2], left, right)
            except:
                index = old
        return left

    return parse_expr()


@dataclass
class EvalContext:
    variables: dict
    functions: dict

    def get_var(self, name):
        return self.variables[name]

    def get_func(self, name):
        return self.functions[name]


def get_default_context():
    import cmath
    import math
    from numbers import Number
    from itertools import chain

    ctx = EvalContext(
        {**{k: v for (k, v) in cmath.__dict__.items() if not k.startswith("_")
            and isinstance(v, Number)}, **{"i": 1j, "j": 1j}},
        {k: v for (k, v) in chain(math.__dict__.items(
        ), cmath.__dict__.items()) if not k.startswith("_") and callable(v)}
    )

    return ctx


defc = get_default_context()


def evalstr(inp: str, parentctx: EvalContext = None):
    if parentctx is None:
        parentctx = defc
    ctx = EvalContext(parentctx.variables.copy(), parentctx.functions.copy())
    while True:
        try:
            tok = tokenize(inp)
            tree = parse(tok)
            print(tree)
            print(tree.write())
            return tree.eval(ctx)
        except UndeclaredVarError as e:
            value = input(f"value for '{e.var}'? ")
            if not value:
                return None
            value = evalstr(value, ctx)
            if not value:
                return None
            ctx.variables[e.var] = value
            continue


while True:
    inp = input("> ")
    if inp == "?":
        print("functions:")
        print("  " + ", ".join(ctx.functions.keys()))
        print("variables:")
        print("  " + ", ".join(ctx.variables.keys()))
    else:
        try:
            res = evalstr(inp)
            if res is None:
                print("-- cancelled")
            else:
                if res.imag == 0:
                    res = res.real
                print(res)
        except ParseError as e:
            print((2 + e.pos) * " " + "↑")
            print(str(e) + " at " + str(e.pos))
        except Exception as e:
            print(str(e))
    print()
