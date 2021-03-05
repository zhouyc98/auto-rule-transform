# Generated from .\RuleCheckTree.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .RuleCheckTreeParser import RuleCheckTreeParser
else:
    from RuleCheckTreeParser import RuleCheckTreeParser

# This class defines a complete generic visitor for a parse tree produced by RuleCheckTreeParser.

class RuleCheckTreeVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by RuleCheckTreeParser#rctree.
    def visitRctree(self, ctx:RuleCheckTreeParser.RctreeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RuleCheckTreeParser#prs.
    def visitPrs(self, ctx:RuleCheckTreeParser.PrsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RuleCheckTreeParser#pr.
    def visitPr(self, ctx:RuleCheckTreeParser.PrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RuleCheckTreeParser#req.
    def visitReq(self, ctx:RuleCheckTreeParser.ReqContext):
        return self.visitChildren(ctx)



del RuleCheckTreeParser