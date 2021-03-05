# Generated from .\RuleCheckTree.g4 by ANTLR 4.8
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\13")
        buf.write("\61\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\3\2\6\2\f\n\2\r\2")
        buf.write("\16\2\r\3\3\3\3\6\3\22\n\3\r\3\16\3\23\3\3\3\3\3\3\3\3")
        buf.write("\3\3\5\3\33\n\3\3\4\3\4\6\4\37\n\4\r\4\16\4 \3\4\3\4\3")
        buf.write("\4\3\4\5\4\'\n\4\3\5\5\5*\n\5\3\5\5\5-\n\5\3\5\3\5\3\5")
        buf.write("\2\2\6\2\4\6\b\2\2\2\65\2\13\3\2\2\2\4\32\3\2\2\2\6&\3")
        buf.write("\2\2\2\b)\3\2\2\2\n\f\5\4\3\2\13\n\3\2\2\2\f\r\3\2\2\2")
        buf.write("\r\13\3\2\2\2\r\16\3\2\2\2\16\3\3\2\2\2\17\33\5\6\4\2")
        buf.write("\20\22\7\3\2\2\21\20\3\2\2\2\22\23\3\2\2\2\23\21\3\2\2")
        buf.write("\2\23\24\3\2\2\2\24\25\3\2\2\2\25\33\5\6\4\2\26\27\5\6")
        buf.write("\4\2\27\30\5\b\5\2\30\33\3\2\2\2\31\33\5\b\5\2\32\17\3")
        buf.write("\2\2\2\32\21\3\2\2\2\32\26\3\2\2\2\32\31\3\2\2\2\33\5")
        buf.write("\3\2\2\2\34\36\7\3\2\2\35\37\5\6\4\2\36\35\3\2\2\2\37")
        buf.write(" \3\2\2\2 \36\3\2\2\2 !\3\2\2\2!\"\3\2\2\2\"#\5\b\5\2")
        buf.write("#\'\3\2\2\2$%\7\3\2\2%\'\5\b\5\2&\34\3\2\2\2&$\3\2\2\2")
        buf.write("\'\7\3\2\2\2(*\7\4\2\2)(\3\2\2\2)*\3\2\2\2*,\3\2\2\2+")
        buf.write("-\7\5\2\2,+\3\2\2\2,-\3\2\2\2-.\3\2\2\2./\7\6\2\2/\t\3")
        buf.write("\2\2\2\t\r\23\32 &),")
        return buf.getvalue()


class RuleCheckTreeParser ( Parser ):

    grammarFileName = "RuleCheckTree.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [  ]

    symbolicNames = [ "<INVALID>", "PROP", "CMP", "ROBJ", "RPROP", "OBJ", 
                      "OTHER", "OTHERS", "CHAR", "NEWLINE" ]

    RULE_rctree = 0
    RULE_prs = 1
    RULE_pr = 2
    RULE_req = 3

    ruleNames =  [ "rctree", "prs", "pr", "req" ]

    EOF = Token.EOF
    PROP=1
    CMP=2
    ROBJ=3
    RPROP=4
    OBJ=5
    OTHER=6
    OTHERS=7
    CHAR=8
    NEWLINE=9

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.8")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class RctreeContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def prs(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RuleCheckTreeParser.PrsContext)
            else:
                return self.getTypedRuleContext(RuleCheckTreeParser.PrsContext,i)


        def getRuleIndex(self):
            return RuleCheckTreeParser.RULE_rctree

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRctree" ):
                listener.enterRctree(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRctree" ):
                listener.exitRctree(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitRctree" ):
                return visitor.visitRctree(self)
            else:
                return visitor.visitChildren(self)




    def rctree(self):

        localctx = RuleCheckTreeParser.RctreeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_rctree)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 9 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 8
                self.prs()
                self.state = 11 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << RuleCheckTreeParser.PROP) | (1 << RuleCheckTreeParser.CMP) | (1 << RuleCheckTreeParser.ROBJ) | (1 << RuleCheckTreeParser.RPROP))) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PrsContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pr(self):
            return self.getTypedRuleContext(RuleCheckTreeParser.PrContext,0)


        def PROP(self, i:int=None):
            if i is None:
                return self.getTokens(RuleCheckTreeParser.PROP)
            else:
                return self.getToken(RuleCheckTreeParser.PROP, i)

        def req(self):
            return self.getTypedRuleContext(RuleCheckTreeParser.ReqContext,0)


        def getRuleIndex(self):
            return RuleCheckTreeParser.RULE_prs

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrs" ):
                listener.enterPrs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrs" ):
                listener.exitPrs(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPrs" ):
                return visitor.visitPrs(self)
            else:
                return visitor.visitChildren(self)




    def prs(self):

        localctx = RuleCheckTreeParser.PrsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_prs)
        try:
            self.state = 24
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 13
                self.pr()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 15 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 14
                        self.match(RuleCheckTreeParser.PROP)

                    else:
                        raise NoViableAltException(self)
                    self.state = 17 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

                self.state = 19
                self.pr()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 20
                self.pr()
                self.state = 21
                self.req()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 23
                self.req()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PrContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PROP(self):
            return self.getToken(RuleCheckTreeParser.PROP, 0)

        def req(self):
            return self.getTypedRuleContext(RuleCheckTreeParser.ReqContext,0)


        def pr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RuleCheckTreeParser.PrContext)
            else:
                return self.getTypedRuleContext(RuleCheckTreeParser.PrContext,i)


        def getRuleIndex(self):
            return RuleCheckTreeParser.RULE_pr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPr" ):
                listener.enterPr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPr" ):
                listener.exitPr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPr" ):
                return visitor.visitPr(self)
            else:
                return visitor.visitChildren(self)




    def pr(self):

        localctx = RuleCheckTreeParser.PrContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_pr)
        self._la = 0 # Token type
        try:
            self.state = 36
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 26
                self.match(RuleCheckTreeParser.PROP)
                self.state = 28 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 27
                    self.pr()
                    self.state = 30 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not (_la==RuleCheckTreeParser.PROP):
                        break

                self.state = 32
                self.req()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 34
                self.match(RuleCheckTreeParser.PROP)
                self.state = 35
                self.req()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ReqContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def RPROP(self):
            return self.getToken(RuleCheckTreeParser.RPROP, 0)

        def CMP(self):
            return self.getToken(RuleCheckTreeParser.CMP, 0)

        def ROBJ(self):
            return self.getToken(RuleCheckTreeParser.ROBJ, 0)

        def getRuleIndex(self):
            return RuleCheckTreeParser.RULE_req

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterReq" ):
                listener.enterReq(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitReq" ):
                listener.exitReq(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitReq" ):
                return visitor.visitReq(self)
            else:
                return visitor.visitChildren(self)




    def req(self):

        localctx = RuleCheckTreeParser.ReqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_req)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 39
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RuleCheckTreeParser.CMP:
                self.state = 38
                self.match(RuleCheckTreeParser.CMP)


            self.state = 42
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==RuleCheckTreeParser.ROBJ:
                self.state = 41
                self.match(RuleCheckTreeParser.ROBJ)


            self.state = 44
            self.match(RuleCheckTreeParser.RPROP)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





