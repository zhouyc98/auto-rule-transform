// Generated from .\RuleCheckTree.g4 by ANTLR 4.8
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class RuleCheckTreeLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		PROP=1, CMP=2, ROBJ=3, RPROP=4, OBJ=5, OTHER=6, OTHERS=7, CHAR=8, SEP=9, 
		NEWLINE=10;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"PROP", "CMP", "ROBJ", "RPROP", "OBJ", "OTHER", "OTHERS", "CHAR", "SEP", 
			"NEWLINE"
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


	public RuleCheckTreeLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "RuleCheckTree.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\f\u0086\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\3\2\3\2\7\2\32\n\2\f\2\16\2\35\13\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2"+
		"\3\3\3\3\7\3(\n\3\f\3\16\3+\13\3\3\3\3\3\3\3\3\3\3\3\3\3\3\4\3\4\7\4\65"+
		"\n\4\f\4\16\48\13\4\3\4\3\4\5\4<\n\4\3\4\3\4\3\4\3\4\3\4\3\4\3\5\3\5\7"+
		"\5F\n\5\f\5\16\5I\13\5\3\5\3\5\5\5M\n\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3"+
		"\6\3\6\7\6X\n\6\f\6\16\6[\13\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\7\3\7"+
		"\7\7g\n\7\f\7\16\7j\13\7\3\7\3\7\3\7\3\7\3\7\3\7\3\b\6\bs\n\b\r\b\16\b"+
		"t\3\b\3\b\3\t\3\t\3\n\6\n|\n\n\r\n\16\n}\3\n\3\n\3\13\5\13\u0083\n\13"+
		"\3\13\3\13\n\33)\66GYht}\2\f\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13"+
		"\25\f\3\2\3\5\2\f\f\17\17aa\2\u0090\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2"+
		"\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23"+
		"\3\2\2\2\2\25\3\2\2\2\3\27\3\2\2\2\5%\3\2\2\2\7\62\3\2\2\2\tC\3\2\2\2"+
		"\13U\3\2\2\2\rd\3\2\2\2\17r\3\2\2\2\21x\3\2\2\2\23{\3\2\2\2\25\u0082\3"+
		"\2\2\2\27\33\7]\2\2\30\32\5\21\t\2\31\30\3\2\2\2\32\35\3\2\2\2\33\34\3"+
		"\2\2\2\33\31\3\2\2\2\34\36\3\2\2\2\35\33\3\2\2\2\36\37\7\61\2\2\37 \7"+
		"r\2\2 !\7t\2\2!\"\7q\2\2\"#\7r\2\2#$\7_\2\2$\4\3\2\2\2%)\7]\2\2&(\5\21"+
		"\t\2\'&\3\2\2\2(+\3\2\2\2)*\3\2\2\2)\'\3\2\2\2*,\3\2\2\2+)\3\2\2\2,-\7"+
		"\61\2\2-.\7e\2\2./\7o\2\2/\60\7r\2\2\60\61\7_\2\2\61\6\3\2\2\2\62\66\7"+
		"]\2\2\63\65\5\21\t\2\64\63\3\2\2\2\658\3\2\2\2\66\67\3\2\2\2\66\64\3\2"+
		"\2\2\679\3\2\2\28\66\3\2\2\29;\7\61\2\2:<\7c\2\2;:\3\2\2\2;<\3\2\2\2<"+
		"=\3\2\2\2=>\7T\2\2>?\7q\2\2?@\7d\2\2@A\7l\2\2AB\7_\2\2B\b\3\2\2\2CG\7"+
		"]\2\2DF\5\21\t\2ED\3\2\2\2FI\3\2\2\2GH\3\2\2\2GE\3\2\2\2HJ\3\2\2\2IG\3"+
		"\2\2\2JL\7\61\2\2KM\7c\2\2LK\3\2\2\2LM\3\2\2\2MN\3\2\2\2NO\7T\2\2OP\7"+
		"r\2\2PQ\7t\2\2QR\7q\2\2RS\7r\2\2ST\7_\2\2T\n\3\2\2\2UY\7]\2\2VX\5\21\t"+
		"\2WV\3\2\2\2X[\3\2\2\2YZ\3\2\2\2YW\3\2\2\2Z\\\3\2\2\2[Y\3\2\2\2\\]\7\61"+
		"\2\2]^\7q\2\2^_\7d\2\2_`\7l\2\2`a\7_\2\2ab\3\2\2\2bc\b\6\2\2c\f\3\2\2"+
		"\2dh\7]\2\2eg\5\21\t\2fe\3\2\2\2gj\3\2\2\2hi\3\2\2\2hf\3\2\2\2ik\3\2\2"+
		"\2jh\3\2\2\2kl\7\61\2\2lm\7Q\2\2mn\7_\2\2no\3\2\2\2op\b\7\2\2p\16\3\2"+
		"\2\2qs\5\21\t\2rq\3\2\2\2st\3\2\2\2tu\3\2\2\2tr\3\2\2\2uv\3\2\2\2vw\b"+
		"\b\2\2w\20\3\2\2\2xy\n\2\2\2y\22\3\2\2\2z|\7a\2\2{z\3\2\2\2|}\3\2\2\2"+
		"}~\3\2\2\2}{\3\2\2\2~\177\3\2\2\2\177\u0080\b\n\2\2\u0080\24\3\2\2\2\u0081"+
		"\u0083\7\17\2\2\u0082\u0081\3\2\2\2\u0082\u0083\3\2\2\2\u0083\u0084\3"+
		"\2\2\2\u0084\u0085\7\f\2\2\u0085\26\3\2\2\2\16\2\33)\66;GLYht}\u0082\3"+
		"\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}