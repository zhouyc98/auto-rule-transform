# coding=utf-8

from antlr4.error.ErrorListener import ErrorListener


class ParserError(Exception):
    pass


class RCTreeErrorListener(ErrorListener):

    def __init__(self, ignore_warning=False, log_fn=None):
        super().__init__()
        self.ignore_warning = ignore_warning
        self.log_fn = print if log_fn is None else log_fn

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if self.ignore_warning:
            self.log_fn("!SyntaxError: " + str(msg))
        else:
            raise ParserError("!SyntaxError: " + str(msg))

    # def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
    #     raise ParserError("!ReportAmbiguity")

    # def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
    #     raise ParserError("!ReportAttemptingFullContext")

    # def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
    #     raise ParserError("!ReportContextSensitivity")
