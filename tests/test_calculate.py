"""L3 计算工具测试 + 安全测试。"""

from agent_cli.tools.builtin.calculate import calculate


class TestBasicArithmetic:
    def test_addition(self):
        assert "= 10" in calculate.invoke({"expression": "3 + 7"})

    def test_subtraction(self):
        assert "= 6" in calculate.invoke({"expression": "10 - 4"})

    def test_multiplication(self):
        assert "= 21" in calculate.invoke({"expression": "3 * 7"})

    def test_division(self):
        assert "= 5.0" in calculate.invoke({"expression": "10 / 2"})

    def test_complex_expression(self):
        assert "= 14" in calculate.invoke({"expression": "2 + 3 * 4"})

    def test_parentheses(self):
        assert "= 20" in calculate.invoke({"expression": "(2 + 3) * 4"})

    def test_float(self):
        assert "6.28" in calculate.invoke({"expression": "3.14 * 2"})

    def test_modulo(self):
        assert "= 1" in calculate.invoke({"expression": "7 % 3"})

    def test_power(self):
        assert "= 8" in calculate.invoke({"expression": "2 ** 3"})

    def test_floor_division(self):
        assert "= 3" in calculate.invoke({"expression": "10 // 3"})

    def test_negative_number(self):
        assert "= -5" in calculate.invoke({"expression": "-5"})

    def test_unary_plus(self):
        assert "= 5" in calculate.invoke({"expression": "+5"})

    def test_math_sqrt(self):
        assert "= 4.0" in calculate.invoke({"expression": "math.sqrt(16)"})

    def test_math_pi(self):
        result = calculate.invoke({"expression": "math.pi"})
        assert "3.14159" in result

    def test_math_sin(self):
        result = calculate.invoke({"expression": "math.sin(0)"})
        assert "= 0.0" in result

    def test_math_pow(self):
        assert "= 8" in calculate.invoke({"expression": "math.pow(2, 3)"})

    def test_allowed_calls(self):
        assert "= 3" in calculate.invoke({"expression": "abs(-3)"})
        assert "= 5" in calculate.invoke({"expression": "max(2, 5)"})
        assert "= 2" in calculate.invoke({"expression": "min(2, 5)"})
        assert "= 10" in calculate.invoke({"expression": "sum([1, 2, 3, 4])"})
        assert "= 4" in calculate.invoke({"expression": "len([1, 2, 3, 4])"})
        assert "= 3" in calculate.invoke({"expression": "round(3.14)"})


class TestErrorHandling:
    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "1 / 0"})
        assert "计算错误" in result

    def test_syntax_error(self):
        result = calculate.invoke({"expression": "1 +"})
        assert "计算错误" in result

    def test_empty_expression(self):
        result = calculate.invoke({"expression": ""})
        assert "计算错误" in result or "计算结果" in result

    def test_non_math_expression(self):
        result = calculate.invoke({"expression": "abc"})
        assert "计算错误" in result


class TestSecurity:
    def test_blocks_import(self):
        result = calculate.invoke({"expression": "__import__('os')"})
        assert "计算错误" in result

    def test_blocks_open(self):
        result = calculate.invoke({"expression": "open('test.txt')"})
        assert "计算错误" in result

    def test_blocks_eval_nesting(self):
        result = calculate.invoke({"expression": "eval('1+1')"})
        assert "计算错误" in result

    def test_blocks_exec(self):
        result = calculate.invoke({"expression": "exec('print(1)')"})
        assert "计算错误" in result

    def test_blocks_builtins_access(self):
        result = calculate.invoke({"expression": "__builtins__"})
        assert "计算错误" in result

    def test_blocks_object_introspection(self):
        # ().__class__.__bases__[0].__subclasses__()  — 经典的 eval 绕过
        result = calculate.invoke({"expression": "().__class__.__bases__[0]"})
        assert "计算错误" in result

    def test_blocks_subclasses_access(self):
        result = calculate.invoke(
            {"expression": "(1).__class__.__bases__[0].__subclasses__()"}
        )
        assert "计算错误" in result

    def test_blocks_globals_access(self):
        result = calculate.invoke({"expression": "().__class__.__init__.__globals__"})
        assert "计算错误" in result

    def test_blocks_getattr_chain(self):
        result = calculate.invoke(
            {"expression": "(1).__class__.__bases__[0].__subclasses__"}
        )
        assert "计算错误" in result

    def test_blocks_lambda(self):
        result = calculate.invoke({"expression": "(lambda: 1)()"})
        assert "计算错误" in result

    def test_blocks_list_comprehension(self):
        result = calculate.invoke({"expression": "[x for x in (1,)]"})
        assert "计算错误" in result

    def test_blocks_dict_comprehension(self):
        result = calculate.invoke({"expression": "{x: x for x in (1,)}"})
        assert "计算错误" in result

    def test_blocks_assignment(self):
        result = calculate.invoke({"expression": "x = 1"})
        assert "计算错误" in result

    def test_blocks_non_math_function(self):
        result = calculate.invoke({"expression": "print(1)"})
        assert "计算错误" in result

    def test_blocks_os_module(self):
        result = calculate.invoke({"expression": "__import__('os').system('ls')"})
        assert "计算错误" in result

    def test_blocks_type_function(self):
        result = calculate.invoke({"expression": "type(1)"})
        assert "计算错误" in result

    def test_blocks_help_function(self):
        result = calculate.invoke({"expression": "help()"})
        assert "计算错误" in result

    def test_blocks_str_format(self):
        result = calculate.invoke({"expression": "'{0}'.format(1)"})
        assert "计算错误" in result

    def test_blocks_f_string(self):
        result = calculate.invoke({"expression": "f'{1}'"})
        assert "计算错误" in result

    def test_blocks_non_math_attribute(self):
        # math 模块的属性白名单限制了可访问的属性
        result = calculate.invoke({"expression": "math.__dict__"})
        assert "计算错误" in result
