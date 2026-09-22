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
        assert "计算错误" in result or result.strip() == "计算结果: __builtins__ = {}"
