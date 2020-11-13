// Generated from .\RuleCheckTree.g4 by ANTLR 4.8
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link RuleCheckTreeParser}.
 */
public interface RuleCheckTreeListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link RuleCheckTreeParser#rctree}.
	 * @param ctx the parse tree
	 */
	void enterRctree(RuleCheckTreeParser.RctreeContext ctx);
	/**
	 * Exit a parse tree produced by {@link RuleCheckTreeParser#rctree}.
	 * @param ctx the parse tree
	 */
	void exitRctree(RuleCheckTreeParser.RctreeContext ctx);
	/**
	 * Enter a parse tree produced by {@link RuleCheckTreeParser#prs}.
	 * @param ctx the parse tree
	 */
	void enterPrs(RuleCheckTreeParser.PrsContext ctx);
	/**
	 * Exit a parse tree produced by {@link RuleCheckTreeParser#prs}.
	 * @param ctx the parse tree
	 */
	void exitPrs(RuleCheckTreeParser.PrsContext ctx);
	/**
	 * Enter a parse tree produced by {@link RuleCheckTreeParser#pr}.
	 * @param ctx the parse tree
	 */
	void enterPr(RuleCheckTreeParser.PrContext ctx);
	/**
	 * Exit a parse tree produced by {@link RuleCheckTreeParser#pr}.
	 * @param ctx the parse tree
	 */
	void exitPr(RuleCheckTreeParser.PrContext ctx);
	/**
	 * Enter a parse tree produced by {@link RuleCheckTreeParser#req}.
	 * @param ctx the parse tree
	 */
	void enterReq(RuleCheckTreeParser.ReqContext ctx);
	/**
	 * Exit a parse tree produced by {@link RuleCheckTreeParser#req}.
	 * @param ctx the parse tree
	 */
	void exitReq(RuleCheckTreeParser.ReqContext ctx);
}