# Docstring Template

Use this template for docstrings across the repository.

## General Guidelines

- Wrap docstring prose at a maximum of 79 characters per line.
- Keep wording precise, non-repetitive, and behavior-focused.
- Use a one-line summary first; add an optional second paragraph when needed.
- `# Notes` should only be included for non-obvious design decisions.
- Format docstrings with `rumdl`.
- When a signature (method/type) exceeds 79 columns, break lines as:

```julia
"""
    method(argument1, argument2, argument3, argument4, argument5, argument6,
        argument7, argument8; kwarg1 = default1, kwarg2 = default2,
        kwarg3 = default3, ...)
"""

"""
    TypeName{T1, T2, T3,
        T4, T5}
"""
```

## Specific Guidelines

### Functions

#### Out-Of-Place / Non-Mutating

Preferred sections:

- `# Arguments`
- `# Keyword Arguments` (when applicable)
- `# Returns`
- `# Throws` (when applicable)
- `# Notes` (non-obvious decisions only)
- `# See also` (optional)

Example:

```julia
"""
    function_name(arg1, arg2; kw1=default)

<One-line behavior-focused summary in present tense>.

<Optional second paragraph for context, assumptions, or
non-obvious behavior>.

# Arguments

- `arg1::Type`: Meaning, units/range, and key constraints.
- `arg2::Type`: Meaning and shape/size expectations.

# Keyword Arguments

- `kw1::Type=default`: Meaning and effect.

# Returns

- `ReturnType`: What is returned, including shape/semantics.

# Throws

- `ExceptionType`: Condition that triggers it.

# Notes

- Assumptions or model-specific conventions.

# See also

- [`related_fn`](@ref), [`other_fn!`](@ref)
"""
function function_name(arg1, arg2; kw1=default)
    ...
end
```

#### In-Place / Mutating

Preferred sections:

- `# Arguments`
- `# Keyword Arguments` (when applicable)
- `# Throws` (when applicable)
- `# Notes` (non-obvious decisions only)

If a mutating method returns `nothing`, omit `# Returns`.
If it returns a value, include `# Returns`.

Example:

```julia
"""
    function_name!(state, input; ...)

Update `<state>` in-place using `<input>`.

# Arguments

- `state::Type`: Mutated object. Required shape/size expectations.
- `input::Type`: Input data contract.

# Keyword Arguments

- `...`

# Throws

- `DimensionMismatch`: When input sizes are incompatible.

# Notes

- In-place contract and required preconditions.
"""
function function_name!(state, input; ...)
    ...
end
```

### Types/Structs

Preferred sections:

- `# Fields`
- `# Notes` (usage constraints or key relationships)

Do not document constructors in the type docstring.

Example:

```julia
"""
    TypeName{...}

<Role of this type in the package>.

# Fields

- `field1::Type`: Meaning.
- `field2::Type`: Meaning.

# Notes

- Usage constraints and relationship to key methods.
"""
struct TypeName{T1, T2}
    ...
end
```

#### Constructors

- Document constructors in their own method docstrings.
- For inner constructors, attach docs explicitly with `@doc """ ... """`
  and define the method on the next line.
- Include summaries in constructors only when behavior is non-obvious
  (validation, coercion, normalization, defaults, special errors, etc.). If
  constructor behavior is straightforward, omit generic summaries like
  `Construct <TypeName> ...`.

Example:

```julia
@doc """
    TypeName(arg1, arg2)

Validate `arg1` and `arg2` and normalize stored coefficients.

# Throws

- `DimensionMismatch`: If `arg1` and `arg2` do not have matching lengths.
"""
function TypeName(arg1, arg2)
    ...
end
```
