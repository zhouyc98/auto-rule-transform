// Generated from .\RuleCheckTree.g4 by ANTLR 4.8
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class RuleCheckTreeParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		PROP=1, CMP=2, ROBJ=3, RPROP=4, OBJ=5, OTHER=6, OTHERS=7, CHAR=8, SEP=9, 
		NEWLINE=10;
	public static final int
		RULE_rctree = 0, RULE_prs = 1, RULE_pr = 2, RULE_req = 3;
	private static String[] makeRuleNames() {
		return new String[] {
			"rctree", "prs", "pr", "req"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "PROP", "CMP", "ROBJ", "RPROP", "OBJ", "OTHER", "OTHERS", "CHAR", 
			"SEP", "NEWLINE"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "RuleCheckTree.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public RuleCheckTreeParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class RctreeContext extends ParserRuleContext {
		public List<PrsContext> prs() {
			return getRuleContexts(PrsContext.class);
		}
		public PrsContext prs(int i) {
			return getRuleContext(PrsContext.class,i);
		}
		public List<TerminalNode> NEWLINE() { return getTokens(RuleCheckTreeParser.NEWLINE); }
		public TerminalNode NEWLINE(int i) {
			return getToken(RuleCheckTreeParser.NEWLINE, i);
		}
		public RctreeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_rctree; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).enterRctree(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).exitRctree(this);
		}
	}

	public final RctreeContext rctree() throws RecognitionException {
		RctreeContext _localctx = new RctreeContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_rctree);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(9); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(8);
				prs();
				}
				}
				setState(11); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << PROP) | (1L << CMP) | (1L << ROBJ) | (1L << RPROP))) != 0) );
			setState(16);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			while ( _alt!=1 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1+1 ) {
					{
					{
					setState(13);
					match(NEWLINE);
					}
					} 
				}
				setState(18);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PrsContext extends ParserRuleContext {
		public PrContext pr() {
			return getRuleContext(PrContext.class,0);
		}
		public List<TerminalNode> PROP() { return getTokens(RuleCheckTreeParser.PROP); }
		public TerminalNode PROP(int i) {
			return getToken(RuleCheckTreeParser.PROP, i);
		}
		public ReqContext req() {
			return getRuleContext(ReqContext.class,0);
		}
		public PrsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_prs; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).enterPrs(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).exitPrs(this);
		}
	}

	public final PrsContext prs() throws RecognitionException {
		PrsContext _localctx = new PrsContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_prs);
		try {
			int _alt;
			setState(30);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(19);
				pr();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(21); 
				_errHandler.sync(this);
				_alt = 1;
				do {
					switch (_alt) {
					case 1:
						{
						{
						setState(20);
						match(PROP);
						}
						}
						break;
					default:
						throw new NoViableAltException(this);
					}
					setState(23); 
					_errHandler.sync(this);
					_alt = getInterpreter().adaptivePredict(_input,2,_ctx);
				} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
				setState(25);
				pr();
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(26);
				pr();
				setState(27);
				req();
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(29);
				req();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PrContext extends ParserRuleContext {
		public TerminalNode PROP() { return getToken(RuleCheckTreeParser.PROP, 0); }
		public ReqContext req() {
			return getRuleContext(ReqContext.class,0);
		}
		public List<PrContext> pr() {
			return getRuleContexts(PrContext.class);
		}
		public PrContext pr(int i) {
			return getRuleContext(PrContext.class,i);
		}
		public PrContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pr; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).enterPr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).exitPr(this);
		}
	}

	public final PrContext pr() throws RecognitionException {
		PrContext _localctx = new PrContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_pr);
		int _la;
		try {
			setState(42);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(32);
				match(PROP);
				setState(34); 
				_errHandler.sync(this);
				_la = _input.LA(1);
				do {
					{
					{
					setState(33);
					pr();
					}
					}
					setState(36); 
					_errHandler.sync(this);
					_la = _input.LA(1);
				} while ( _la==PROP );
				setState(38);
				req();
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(40);
				match(PROP);
				setState(41);
				req();
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ReqContext extends ParserRuleContext {
		public TerminalNode RPROP() { return getToken(RuleCheckTreeParser.RPROP, 0); }
		public TerminalNode CMP() { return getToken(RuleCheckTreeParser.CMP, 0); }
		public TerminalNode ROBJ() { return getToken(RuleCheckTreeParser.ROBJ, 0); }
		public ReqContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_req; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).enterReq(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof RuleCheckTreeListener ) ((RuleCheckTreeListener)listener).exitReq(this);
		}
	}

	public final ReqContext req() throws RecognitionException {
		ReqContext _localctx = new ReqContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_req);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(45);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==CMP) {
				{
				setState(44);
				match(CMP);
				}
			}

			setState(48);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==ROBJ) {
				{
				setState(47);
				match(ROBJ);
				}
			}

			setState(50);
			match(RPROP);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\f\67\4\2\t\2\4\3"+
		"\t\3\4\4\t\4\4\5\t\5\3\2\6\2\f\n\2\r\2\16\2\r\3\2\7\2\21\n\2\f\2\16\2"+
		"\24\13\2\3\3\3\3\6\3\30\n\3\r\3\16\3\31\3\3\3\3\3\3\3\3\3\3\5\3!\n\3\3"+
		"\4\3\4\6\4%\n\4\r\4\16\4&\3\4\3\4\3\4\3\4\5\4-\n\4\3\5\5\5\60\n\5\3\5"+
		"\5\5\63\n\5\3\5\3\5\3\5\3\22\2\6\2\4\6\b\2\2\2<\2\13\3\2\2\2\4 \3\2\2"+
		"\2\6,\3\2\2\2\b/\3\2\2\2\n\f\5\4\3\2\13\n\3\2\2\2\f\r\3\2\2\2\r\13\3\2"+
		"\2\2\r\16\3\2\2\2\16\22\3\2\2\2\17\21\7\f\2\2\20\17\3\2\2\2\21\24\3\2"+
		"\2\2\22\23\3\2\2\2\22\20\3\2\2\2\23\3\3\2\2\2\24\22\3\2\2\2\25!\5\6\4"+
		"\2\26\30\7\3\2\2\27\26\3\2\2\2\30\31\3\2\2\2\31\27\3\2\2\2\31\32\3\2\2"+
		"\2\32\33\3\2\2\2\33!\5\6\4\2\34\35\5\6\4\2\35\36\5\b\5\2\36!\3\2\2\2\37"+
		"!\5\b\5\2 \25\3\2\2\2 \27\3\2\2\2 \34\3\2\2\2 \37\3\2\2\2!\5\3\2\2\2\""+
		"$\7\3\2\2#%\5\6\4\2$#\3\2\2\2%&\3\2\2\2&$\3\2\2\2&\'\3\2\2\2\'(\3\2\2"+
		"\2()\5\b\5\2)-\3\2\2\2*+\7\3\2\2+-\5\b\5\2,\"\3\2\2\2,*\3\2\2\2-\7\3\2"+
		"\2\2.\60\7\4\2\2/.\3\2\2\2/\60\3\2\2\2\60\62\3\2\2\2\61\63\7\5\2\2\62"+
		"\61\3\2\2\2\62\63\3\2\2\2\63\64\3\2\2\2\64\65\7\6\2\2\65\t\3\2\2\2\n\r"+
		"\22\31 &,/\62";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}