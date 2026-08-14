"""
Arithmetic expressions which can be converted from and to a vaguely
Lispish textual representation (hence "Sexpr" as the base class)
and which can additionally be converted (one way) to PyArrow compute
expressions so they can be evaluated against a table.
"""

import dataclasses
import json
import re

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Iterable, Mapping

import pyarrow.compute as pc

__all__ = [
    "Sexpr", "Constant", "Column", "Op",
    "parse_sexps",
]

#
# Public expression node classes
#

class Sexpr(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_arrow(self) -> pc.Expression:
        raise NotImplementedError


@dataclasses.dataclass(slots=True)
class Constant(Sexpr):
    val: str | int | float | bool | None

    def __str__(self) -> str:
        # we use JSON notation for constants, including booleans
        # and None, because that's easy to generate on the JavaScript
        # side; conveniently, that means we can do this to get an
        # unambiguous representation of any constant:
        return json.dumps(self.val)

    def to_arrow(self) -> pc.Expression:
        return pc.scalar(self.val)

@dataclasses.dataclass(slots=True)
class Column(Sexpr):
    colname: str

    def __str__(self) -> str:
        # we use the Common Lisp "self-quoting symbol" notation, :colname,
        # for columns; this cannot collide with any JSON scalar value
        return f":{self.colname}"

    def to_arrow(self) -> pc.Expression:
        return pc.field(self.colname)

@dataclasses.dataclass(slots=True)
class Op(Sexpr):
    opname: str
    args: list[Sexpr]
    kwargs: dict[str, Sexpr]

    def __str__(self) -> str:
        # kwargs use the historical alternative self-quoting symbol
        # notation, &argname, so they can't collide with column names
        return (
            f"({self.opname} "
            + " ".join(f"&{key} {val}" for key, val in self.kwargs.items())
            + " ".join(str(arg) for arg in self.args)
            + ")"
        )

    def to_arrow(self) -> pc.Expression:
        spec = PYARROW_OPS.get(self.opname)
        if spec is None:
            raise ValueError(
                f"operation {self.opname} does not correspond"
                f" to an Arrow compute function"
            )

        spec.check_arity(self.opname, len(self.args))
        spec.check_kwarg_names(self.opname, self.kwargs.keys())
        return getattr(pc, spec.arrow_name)(
            *spec.convert_pargs(self.args),
            **spec.convert_kwargs(self.opname, self.kwargs)
        )


#
# Parsing
#

SEXP_TOKEN_RE = re.compile(r"""
\s*
(?>
  (?P<constant> (?:null|true|false)\b
  |  " (?> \\ (?>["\\/bfnrt] | u[a-fA-F0-9]{4}) | [^"\\\0-\x1F\x7F]+ )* "
  | -? (?> 0 | [1-9][0-9]* ) (?> \. [0-9]+ )? (?> [eE][+-]?[0-9]+ )?
  )
| (?P<symbol> [:&]? (?> \w+\b ) )
| (?P<punct> [()] )  # currently (a . b) and 'x are not used
| (?P<error> \S )
)
""", re.S | re.U | re.X)


@dataclasses.dataclass(slots=True)
class OpUnderConstruction:
    """parser stack entry, will become an Op when closed"""
    name: str | None = None
    args: list[Sexpr] = dataclasses.field(default_factory=list)
    kwargs: dict[str, Sexpr] = dataclasses.field(default_factory=dict)
    pending_key: str | None = None

    def append(self, val: Sexpr | str) -> None:
        if isinstance(val, Sexpr):
            if self.name is None:
                # ??? we might need lists that aren't unevaluated forms, not sure yet
                raise ValueError(f"sexp parse error: '{val}' cannot be first element of a form")
            if self.pending_key is None:
                self.args.append(val)
            else:
                self.kwargs[self.pending_key] = val
                self.pending_key = None
        else:
            assert isinstance(val, str)
            if self.name is None:
                if val[0] in (':', '&'):
                    raise ValueError(f"sexp parse error: '{val}' cannot be first element of a form")
                self.name = val
            else:
                assert val[0] != ':'  # :foo should already be Column("foo")
                if val[0] != '&':
                    raise ValueError(f"sexp parse error: '{val}' can only be the first element of a form")
                if self.pending_key is not None:
                    raise ValueError(f"sexp parse error: no value given for named arg '&{self.pending_key}'")
                key = val[1:]
                if key in self.kwargs:
                    raise ValueError(f"sexp parse error: duplicate named arg '&{key}'")
                self.pending_key = key

    def finish(self) -> Op:
        if self.name is None:
            # () is *not* understood as another way to write null
            raise ValueError("sexp parse error: empty form")
        if self.pending_key:
            raise ValueError(f"sexp parse error: no value given for named arg '&{self.pending_key}'")
        return Op(self.name, self.args, self.kwargs)


def parse_sexps(text: str) -> Iterable[Sexpr]:
    "Parse TEXT as a sequence of Sexprs."
    stack: list[OpUnderConstruction] = []

    for m in SEXP_TOKEN_RE.finditer(text):
        if (tok := m.group("constant")) is not None:
            assert m.group("symbol") is None
            assert m.group("punct") is None
            assert m.group("error") is None

            lit = Constant(json.loads(tok))
            if stack:
                stack[-1].append(lit)
            else:
                yield lit

        elif (tok := m.group("symbol")) is not None:
            assert m.group("punct") is None
            assert m.group("error") is None

            if tok[0] == ':':
                lit = Column(tok[1:])
                if stack:
                    stack[-1].append(lit)
                else:
                    yield lit

            else:
                if not stack:
                    raise ValueError(f"sexp parse error: symbol '{tok}' not within any list")
                stack[-1].append(tok)

        elif (tok := m.group("punct")) is not None:
            assert m.group("error") is None
            if tok == '(':
                stack.append(OpUnderConstruction())
            elif tok == ')':
                if not stack:
                    raise ValueError("sexp parse error: unmatched ')'")
                op = stack.pop().finish()
                if stack:
                    stack[-1].append(op)
                else:
                    yield op
            else:
                raise AssertionError(f"unsupported punct token '{tok!r}'")

        else:
            tok = m.group("error")
            assert tok is not None
            raise ValueError(f"sexp parse error: character '{tok!r}' cannot appear outside strings")

    if stack:
        raise ValueError(f"sexp parse error: {len(stack)} unclosed '(' at EOF")


#
# Kwarg converter functions.
# Positional arguments to Arrow compute functions can always be passed as
# simply <sexpr>.to_arrow(), but keyword arguments sometimes need to be
# passed as unboxed values instead, or otherwise massaged.
#
def kwarg_as_arrow(opname: str, kw: str, arg: Sexpr) -> pc.Expression:
    try:
        return arg.to_arrow()
    except Exception as e:
        raise ValueError(
            f"calling {opname}: invalid value for {kw}: {e}"
        ) from e


def kwarg_as_bool(opname: str, kw: str, arg: Sexpr) -> bool:
    if not isinstance(arg, Constant):
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg} is not a constant"
        )
    v = arg.val
    if not isinstance(v, bool):
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg} is not a boolean"
        )
    return v


def kwarg_as_nat(opname: str, kw: str, arg: Sexpr) -> int:
    if not isinstance(arg, Constant):
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg} is not a constant"
        )
    v = arg.val
    if not isinstance(v, int) or v < 0:
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg} is not an integer >= 0"
        )
    return v


def kwarg_as_str(opname: str, kw: str, arg: Sexpr) -> str:
    if not isinstance(arg, Constant):
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg} is not a constant"
        )
    v = arg.val
    if not isinstance(v, str):
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg!r} is not a string"
        )
    return v


def kwarg_as_choice(
    opname: str, kw: str, arg: Sexpr,
    *,
    choices: Iterable[str]
) -> str:
    v = kwarg_as_str(opname, kw, arg)
    if v not in choices:
        raise ValueError(
            f"calling {opname}: invalid value for {kw}:"
            f" {arg!r} is none of: {', '.join(choices)}"
        )
    return v


@dataclasses.dataclass(frozen=True, slots=True)
class ArrowComputeFnSpec:
    """Describes how to synthesize a call to an Arrow compute function.

    To do this, we need to know the compute function's name, the number of positional
    arguments it takes, and what its keyword-only arguments are.  (All keyword arguments
    to pyarrow compute functions are keyword-only.  We do not bother tracking which
    keyword arguments are mandatory, as Arrow will throw an exception if mandatory keyword
    arguments are missing.)
    """
    arrow_name: str
    arity: int | slice
    kwarg_converters: dict[str, Callable[[str, str, Sexpr], Any]]

    def check_arity(self, opname: str, actual: int) -> None:
        """Raise an exception if ACTUAL is an incorrect number of
        actual arguments to this compute function.
        """

        if not isinstance(actual, int):
            raise TypeError(f"actual shouldn't be a {type(actual).__name__}")
        if actual < 0:
            raise ValueError(f"{actual!r} actual arguments is impossible")

        if isinstance(self.arity, int):
            if self.arity != actual:
                raise ValueError(
                    f"operation {opname} requires {self.arity} args,"
                    f" have {actual}"
                )
            return

        if isinstance(self.arity, slice):
            min_args = self.arity.start
            if min_args is not None and actual < min_args:
                raise ValueError(
                    f"operation {opname} requires at least {min_args}"
                    f" args, have {actual}"
                )
            # we're using a slice object to hold the min/max but it
            # makes more sense in this context for max to be inclusive
            max_args = self.arity.stop
            if max_args is not None and actual > max_args:
                raise ValueError(
                    f"operation {opname} requires no more than {max_args}"
                    f" args, have {actual}"
                )
            return

        raise AssertionError(
            f"bad spec: arity shouldn't be a {type(self.arity).__name__}"
        )

    def check_kwarg_names(self, opname: str, actual: Iterable[str]) -> None:
        """Raise an exception if any of the actual keyword argument names
        in ACTUAL aren't valid keyword arguments to this compute function."""
        if (bad_kwargs := sorted(
            arg for arg in actual
            if arg not in self.kwarg_converters
        )):
            s = "s" if len(bad_kwargs) > 1 else ""
            raise ValueError(
                f"invalid keyword argument{s} to {opname}: "
                + ", ".join(bad_kwargs)
            )

    def convert_pargs(self, args: Iterable[Sexpr]) -> tuple[pc.Expression]:
        """Convert all positional arguments to the form expected by this
        compute function."""
        return tuple(arg.to_arrow() for arg in args)

    def convert_kwargs(
        self, opname: str, kwargs: Mapping[str, Sexpr]
    ) -> dict[str, Any]:
        """Convert all keyword arguments to the form expected by this
        compute function."""
        return {
            kw: self.kwarg_converters[kw](opname, kw, val)
            for kw, val in kwargs.items()
        }


# Shorthands for use in PYARROW_OPS
A = kwarg_as_arrow
B = kwarg_as_bool
N = kwarg_as_nat
S = kwarg_as_str

CM = partial(kwarg_as_choice, choices=("only_valid", "only_null", "all"))
CN = partial(kwarg_as_choice, choices=("NFC", "NFKC", "NFD", "NFKD"))
CR = partial(kwarg_as_choice,
             choices=("half_to_even", "half_to_odd", "half_down", "half_up",
                      "half_towards_zero", "half_towards_infinity",
                      "down", "up", "towards_zero", "towards_infinity"))
CU = partial(kwarg_as_choice, choices=("ignore", "raise"))

def F(
    name: str,
    arity: int | slice,
    **kwarg_converters: Callable[[str, str, Sexpr], Any],
) -> ArrowComputeFnSpec:
    return ArrowComputeFnSpec(name, arity, kwarg_converters)


# The mapping from opnames to pyarrow compute functions.
#
# Our opnames mostly match pyarrow's names, except:
#  - whenever there's both "arithop" and "arithop_checked", we expose
#    only one of the two (usually _checked, but for elementary functions
#    we prefer the versions that return NaN)
#  - many of the string operations have been renamed for clarity and
#    concision; notably, "utf8_" prefixes have been removed,
#    and the "binary_" prefix changed to "byte_"
#  - trailing underscores to avoid Python keywords (and_, or_, not_, etc)
#    are elided
#
# Also, some compute functions are just not supported at all:
#
#   - pivot_wider, quantile, tdigest (need array literal arguments)
#   - case_when (needs "struct of boolean arguments" literals and also I don't understand how to use it)
#   - ascii_reverse, ascii_center, ascii_lpad, ascii_rpad, ascii_ltrim, ascii_rtrim, ascii_trim
#     (redundant to the Unicode-aware versions)
#   - binary_reverse (only works on byte strings which we don't expose)
#   - binary_join, binary_join_element_wise (I can't figure out how to call these correctly,
#                                            I get ArrowNotImplementedError for everything)
#
#   - the date and time functions (not relevant to this application)
#
# We do not expose the 'options' or 'memory_pool' arguments to any compute function.
PYARROW_OPS = {
    # Element-wise operations:
    # Arithmetic
    "negate":      F("negate_checked", 1),      # -x
    "abs":         F("abs_checked", 1),         # |x|
    "sign":        F("sign", 1),                # sgn x

    "add":         F("add_checked", 2),         # x + y
    "subtract":    F("subtract_checked", 2),    # x - y
    "multiply":    F("multiply_checked", 2),    # x * y
    "divide":      F("divide_checked", 2),      # x / y

    "bit_not":     F("bit_wise_not", 1),        # ~x      (bitwise)
    "bit_and":     F("bit_wise_and", 2),        #  x &  y (bitwise)
    "bit_or":      F("bit_wise_or", 2),         #  x |  y (bitwise)
    "bit_xor":     F("bit_wise_xor", 2),        #  x ^  y (bitwise)
    "shift_left":  F("shift_left_checked", 2),  #  x << y
    "shift_right": F("shift_right_checked", 2), #  x >> y

    "max_element_wise": F("max_element_wise", slice(1, None), skip_nulls=B),   # For each element, maximum of all args.
    "min_element_wise": F("max_element_wise", slice(1, None), skip_nulls=B),   # For each element, minimum of all args.

    # Rounding
    "ceil":  F("ceil", 1),      # Smallest integer not less than x (round toward positive infinity)
    "floor": F("floor", 1),     # Greatest integer not more than x (round toward negative infinity)
    "trunc": F("trunc", 1),     # Take the integral part (truncate towards zero)

    # round to 'ndigits' decimal places
    "round": F("round", 1, ndigits=N, round_mode=CR),
    # Same as 'round' but ndigits is a second positional arg, which can be an array
    "round_binary": F("round_binary", 2, round_mode=CR),
    # Round to nearest multiple of 'multiple' ('nearest' defined by round_mode)
    "round_to_multiple": F("round_to_multiple", 2, multiple=A, round_mode=CR),

    # Powers, exponentials, and logarithms; we only expose the NaN-producing variants
    # (there is no "power" function that produces Inf and/or NaN for all error cases,
    # so we expose the one that *doesn't* wrap on integer overflow)
    "power": F("power_checked", 2), # x^{y}
    "sqrt":  F("sqrt", 1),          # √x
    "hypot": F("hypot", 2),         # √(x² + y²)
    "exp":   F("exp", 1),           # e^{x}
    "expm1": F("expm1", 1),         # e^{x} - 1
    "ln":    F("ln", 1),            # log_{e}  x
    "log1p": F("log1p", 1),         # log_{e}  (1+x)
    "log10": F("log10", 1),         # log_{10} x
    "log2":  F("log2", 1),          # log_{2}  x
    "logb":  F("logb", 2),          # log_{b}  x  ; args (x, b)

    # Trigonometry; likewise
    "sin":   F("sin", 1),           # sin x
    "cos":   F("cos", 1),           # cos x
    "tan":   F("tan", 1),           # tan x
    "acos":  F("acos", 1),          # cos^{-1} x
    "asin":  F("asin", 1),          # sin^{-1} x
    "atan":  F("atan", 1),          # tan^{-1} x
    "atan2": F("atan2", 2),         # tan^{-1} (x/y)  ; signs control quadrant

    # Hyperbolic trigonometry
    "sinh":  F("sinh", 1),          # sinh x
    "cosh":  F("cosh", 1),          # cosh x
    "tanh":  F("tanh", 1),          # tanh x
    "asinh": F("asinh", 1),         # sinh^{-1} x
    "acosh": F("acosh", 1),         # cosh^{-1} x
    "atanh": F("atanh", 1),         # tanh^{-1} x

    # Element-wise string operations
    "lower":            F("utf8_lower", 1),              # Change all characters to lowercase.
    "upper":            F("utf8_upper", 1),              # Change all characters to uppercase.
    "swapcase":         F("utf8_swapcase", 1),           # Change lowercase characters to uppercase and uppercase characters to lowercase.
    "capitalize":       F("utf8_capitalize", 1),         # Change the first character to uppercase.
    "title":            F("utf8_title", 1),              # Change the first character of each word to uppercase.
    "reverse":          F("utf8_reverse", 1),            # Reverse codepoint by codepoint.
    "repeat":           F("binary_repeat", 2),           # For each pair (string, count), generate "stringstringstring..." with count repeats.
    "normalize":        F("utf8_normalize", 1, form=CN), # Convert to Unicode normalization form 'form' (e.g. NFC).

    "ascii_lower":      F("ascii_lower", 1),             # Change ASCII characters to lowercase; non-ASCII characters are unchanged.
    "ascii_upper":      F("ascii_upper", 1),             # Change ASCII characters to uppercase.
    "ascii_capitalize": F("ascii_capitalize", 1),        # Change the first character to uppercase if it is ASCII lowercase.
    "ascii_title":      F("ascii_title", 1),             # Titlecase each ASCII word.
    "ascii_swapcase":   F("ascii_swapcase", 1),          # Change ASCII lowercase characters to uppercase and ASCII uppercase characters to lowercase.

    "length":           F("utf8_length", 1),             # Length of string in Unicode codepoints (*not* graphemes)
    "byte_length":      F("binary_length", 1),           # Length of string in bytes of UTF-8 storage representation.

    # Replace a *byte* slice of each string, from the start position to the stop position,
    # with the bytes of the replacement.  Stop positions are exclusive (like Python [start:stop])
    "byte_replace_slice": F("binary_replace_slice", 1, start=N, stop=N, replacement=S),

    # Replace a *codepoint* slice of each string, from the start position to the stop position,
    # with the codepoints of the replacement.  Stop positions are exclusive (like Python [start:stop])
    "replace_slice": F("utf8_replace_slice", 1, start=N, stop=N, replacement=S),

    # Replace all non-overlapping occurrences of the literal string 'pattern' with 'replacement'.
    "replace_substring": F("replace_substring", 1, pattern=S, replacement=S),

    # Replace all non-overlapping occurrences of the regular expression 'pattern' with 'replacement'.
    # Based on an offhand mention on a different page of the Arrow documentation, the
    # regex syntax is _probably_ RE2: https://github.com/google/re2/wiki/Syntax
    # I don't know whether there is any way to insert captured text into the replacement.
    "replace_substring_regex": F("replace_substring_regex", 1, pattern=S, replacement=S),

    # Pad strings to width 'width' with padding character 'padding'.  'width' is a
    # mandatory argument; 'padding' defaults to SPACE, except for 'zero_fill', where it
    # defaults to zero; 'lean_left_on_odd_padding' defaults to True.
    # The original text is placed...
    "rpad":      F("utf8_rpad",      1, width=N, padding=S),  # at the far left of the new string.
    "zero_fill": F("utf8_zero_fill", 1, width=N, padding=S),  # at the far left, preserving leading sign characters.
    "lpad":      F("utf8_lpad",      1, width=N, padding=S),  # at the far right of the new string.
    "center":    F("utf8_center",    1, width=N, padding=S, lean_left_on_odd_padding=B),  # in the middle of the new string.

    "ltrim":                  F("utf8_ltrim", 1, characters=S), # Trim leading characters in 'characters'
    "rtrim":                  F("utf8_rtrim", 1, characters=S), # Trim trailing characters in 'characters'
    "trim":                   F("utf8_trim", 1, characters=S),  # Trim leading and trailing characters in 'characters'
    "ltrim_whitespace":       F("utf8_ltrim_whitespace", 1),    # Trim leading Unicode whitespace characters.
    "rtrim_whitespace":       F("utf8_rtrim_whitespace", 1),    # Trim trailing Unicode whitespace characters.
    "trim_whitespace":        F("utf8_trim_whitespace", 1),     # Trim leading and trailing Unicode whitespace characters.
    "ltrim_ascii_whitespace": F("ascii_ltrim_whitespace", 1),   # Trim leading ASCII whitespace characters.
    "rtrim_ascii_whitespace": F("ascii_rtrim_whitespace", 1),   # Trim trailing ASCII whitespace characters.
    "trim_ascii_whitespace":  F("ascii_trim_whitespace", 1),    # Trim leading and trailing ASCII whitespace characters.

    "split":                  F("split_pattern", 1, pattern=S, max_splits=N, reverse=B),       # Split string according to separator.
    "split_regex":            F("split_pattern_regex", 1, pattern=S, max_splits=N, reverse=B), # Split string according to regex pattern.
    "split_whitespace":       F("utf8_split_whitespace", 1, max_splits=N, reverse=B),          # Split string at Unicode whitespace.
    "split_ascii_whitespace": F("ascii_split_whitespace", 1, max_splits=N, reverse=B),         # Split string at ASCII whitespace.

    # Extract the substrings captured by the _named_ capture patterns in 'pattern'.
    # It is an _error_ if 'pattern' contains any anonymous capture patterns (non-capturing
    # grouping parentheses are OK).
    "extract_regex": F("extract_regex", 1, pattern=S),
    # Same as extract_regex, but returns indices of the first and last character in each
    # capture (note: this is not what the documentation says it does; also, the index of
    # the last character is one less than what you'd want to extract the capture from the
    # original string with a slice expression).
    "extract_regex_span": F("extract_regex_span", 1, pattern=S),

    # take the slice [start:stop:step] of each string, measured in...
    "slice":      F("utf8_slice_codeunits", 1, start=N, stop=N, step=N), # Unicode codepoints
    "byte_slice": F("binary_slice", 1, start=N, stop=N, step=N),         # bytes

    # Substring matching
    "count_substring":       F("count_substring",       1, pattern=S, ignore_case=B), # Count occurrences of literal 'pattern'
    "count_substring_regex": F("count_substring_regex", 1, pattern=S, ignore_case=B), # Count matches of regex 'pattern'
    "find_substring":        F("find_substring",        1, pattern=S, ignore_case=B), # Index of first occurrence of literal 'pattern', -1 if not found
    "find_substring_regex":  F("find_substring_regex",  1, pattern=S, ignore_case=B), # Index of first match of regex 'pattern', -1 if not found
    "match_substring":       F("match_substring",       1, pattern=S, ignore_case=B), # True for each string containing literal 'pattern'
    "match_substring_regex": F("match_substring_regex", 1, pattern=S, ignore_case=B), # True for each string containing match for regex 'pattern'
    "match_like":            F("match_like",            1, pattern=S, ignore_case=B), # True for each string containing match for SQL-style LIKE 'pattern'.
    "ends_with":             F("ends_with",             1, pattern=S, ignore_case=B), # True for each string ending with literal 'pattern'
    "starts_with":           F("starts_with",           1, pattern=S, ignore_case=B), # True for each string starting with literal 'pattern'


    # Cumulative operations
    "cumulative_sum":  F("cumulative_sum_checked",  1, start=A, skip_nulls=B),     # Cumulative sum
    "cumulative_prod": F("cumulative_prod_checked", 1, start=A, skip_nulls=B),     # Cumulative product
    "cumulative_max":  F("cumulative_max",          1, start=A, skip_nulls=B),     # Cumulative maximum
    "cumulative_min":  F("cumulative_min",          1, start=A, skip_nulls=B),     # Cumulative minimum
    "cumulative_mean": F("cumulative_mean",         1, start=A, skip_nulls=B),     # Cumulative arithmetic mean

    # Aggregating operations
    "count":              F("count",              1, mode=CM),                             # Count of null / non-null values.
    "count_distinct":     F("count_distinct",     1, mode=CM),                             # Count of unique values.
    "index":              F("index",              2, start=N, end=N),                      # Index of the first occurrence of a given value.

    "first":              F("first",              1, skip_nulls=B, min_count=N),           # First element in each group.
    "last":               F("last",               1, skip_nulls=B, min_count=N),           # Last element in each group.
    "first_last":         F("first_last",         1, skip_nulls=B, min_count=N),           # First and last elements.
    "all":                F("all",                1, skip_nulls=B, min_count=N),           # True if all elements are true.
    "any":                F("any",                1, skip_nulls=B, min_count=N),           # True if any element is true.
    "sum":                F("sum",                1, skip_nulls=B, min_count=N),           # Sum of elements.
    "product":            F("product",            1, skip_nulls=B, min_count=N),           # Product of elements.
    "max":                F("max",                1, skip_nulls=B, min_count=N),           # Find minimum element.
    "min":                F("min",                1, skip_nulls=B, min_count=N),           # Find maximum element.
    "min_max":            F("min_max",            1, skip_nulls=B, min_count=N),           # Find minimum and maximum

    "mean":               F("mean",               1, skip_nulls=B, min_count=N),           # Arithmetic mean.
    "approximate_median": F("approximate_median", 1, skip_nulls=B, min_count=N),           # Approximate median.
    "mode":               F("mode",               1, skip_nulls=B, min_count=N),           # Modal (most common) value.

    "skew":               F("skew",               1, skip_nulls=B, min_count=N, biased=B), # Skewness.
    "kurtosis":           F("kurtosis",           1, skip_nulls=B, min_count=N, biased=B), # Kurtosis.
    "variance":           F("variance",           1, skip_nulls=B, min_count=N, ddof=N),   # Variance.
    "stddev":             F("stddev",             1, skip_nulls=B, min_count=N, ddof=N),   # Standard deviation.

    # Numeric categorization: Produce boolean vector which is true for each element that is...
    "is_finite":          F("is_finite", 1),              # finite
    "is_inf":             F("is_inf", 1),                 # infinity
    "is_nan":             F("is_nan", 1),                 # NaN
    "is_null":            F("is_null", 1, nan_is_null=B), # null (and optionally NaN)
    "is_valid":           F("is_valid", 1),               # non-null
    "true_unless_null":   F("true_unless_null", 1),       # non-null; null produces null, not false

    # String categorization: Produce boolean vector which is true for each element that's entirely...
    "is_alnum":           F("utf8_is_alnum", 1),          # alphanumeric (gencat L* + N*)
    "is_alpha":           F("utf8_is_alpha", 1),          # alphabetic   (gencat L*)
    "is_ascii":           F("string_is_ascii", 1),        # ASCII        (all code points <= U+007F)
    "is_decimal":         F("utf8_is_decimal", 1),        # decimal      (gencat Nd only)
    "is_digit":           F("utf8_is_digit", 1),          # digits       (gencat Nd and No, but *not* Nl)
    "is_lower":           F("utf8_is_lower", 1),          # lowercase    (gencat Ll, approximately)
    "is_numeric":         F("utf8_is_numeric", 1),        # numeric      (gencat N*)
    "is_printable":       F("utf8_is_printable", 1),      # printable    (not gencat C*)
    "is_space":           F("utf8_is_space", 1),          # whitespace   (gencat Z* or bidi class WS, B, S)
    "is_upper":           F("utf8_is_upper", 1),          # uppercase    (gencat Lu, approximately)
    "is_title":           F("utf8_is_title", 1),          # titlecase    (Like These Words, Using Gencats Ll And Lu)
    "ascii_is_alnum":     F("ascii_is_alnum", 1),         # ASCII alphanumeric (A-Za-z0-9)
    "ascii_is_alpha":     F("ascii_is_alpha", 1),         # ASCII alphabetic   (A-Za-z)
    "ascii_is_decimal":   F("ascii_is_decimal", 1),       # ASCII decimal      (0-9)
    "ascii_is_lower":     F("ascii_is_lower", 1),         # ASCII lowercase    (a-z)
    "ascii_is_printable": F("ascii_is_printable", 1),     # ASCII printable    (U+0020 .. U+007E)
    "ascii_is_space":     F("ascii_is_space", 1),         # ASCII whitespace   (HT LF VT FF CR SPC)
    "ascii_is_upper":     F("ascii_is_upper", 1),         # ASCII uppercase    (A-Z)
    "ascii_is_title":     F("ascii_is_title", 1),         # ASCII titlecase    (Like These Words, Using a-z And A-Z)

    # Reverse array indexing
    "is_in":           F("is_in", 1, value_set=A, skip_nulls=B),    # True for each element that is an element of 'value_set'.
    "index_in":        F("index_in", 1, value_set=A, skip_nulls=B), # Index in 'value_set' of each element that is present; absent values return null
    "indices_nonzero": F("indices_nonzero", 1),                     # Return indices of all array elements that are not zero, false, or null.

    # Comparison
    "equal":         F("equal", 2),         # x == y
    "greater":       F("greater", 2),       # x >  y  (ordered)
    "greater_equal": F("greater_equal", 2), # x >= y  (ordered)
    "less":          F("less", 2),          # x <  y  (ordered)
    "less_equal":    F("less_equal", 2),    # x <= y  (ordered)
    "not_equal":     F("not_equal", 2),     # x != y

    # Logical operators
    # The bare functions uniformly propagate null: e.g. null ∧ false = null.
    # "Kleene" functions mask null when the other arg can determine the
    # result: e.g. null k∧ false = false.  Neither has "short circuit"
    # behavior, and the truth tables are all symmetric (except for and_not).
    "invert":         F("invert", 1),         # ¬x
    "and":            F("and_", 2),           #  x ∧  y
    "and_kleene":     F("and_kleene", 2),     #  x ∧  y (Kleene)
    "and_not":        F("and_not", 2),        #  x ∧ ¬y
    "and_not_kleene": F("and_not_kleene", 2), #  x ∧ ¬y (Kleene)
    "or":             F("or_", 2),            #  x ∨  y
    "or_kleene":      F("or_kleene", 2),      #  x ∨  y (Kleene)
    "xor":            F("xor", 2),            #  x ⊻  y

    # Selecting / Multiplexing
    "choose":   F("choose", slice(2, None)),   # Choose values from several arrays.
    "coalesce": F("coalesce", slice(1, None)), # Select the first non-null value.
    "if_else":  F("if_else", 3),               # Take arg 2 if arg 1 is true, else arg 3.

    # Conversions
    "cast": F("cast", 1, target_type=S, safe=B), # Cast array values to another data type.
}
