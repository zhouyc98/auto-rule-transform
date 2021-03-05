grammar RuleCheckTree;

/* Parser Rules */
//prs: all p-r pattern: ((())], (((())]], [(()))
//pr:  complete p-r pattern: ((()()))()
//RCTree
rctree: prs+;

//all pr pattern
prs:    pr
    |   PROP+ pr
    |   pr req
    |   req
;

//prop-req pattern
pr:     PROP pr+ req
    |   PROP req
;

//requirement
req: CMP? ROBJ? RPROP;


/* Lexer Rules*/
//non-greedy
PROP:   '[' CHAR*? '/prop]';
CMP:    '[' CHAR*? '/cmp]';
ROBJ:   '[' CHAR*? '/' 'a'? 'Robj]';
RPROP:  '[' CHAR*? '/' 'a'? 'Rprop]';
OBJ:    '[' CHAR*? '/obj]' ->skip;
OTHER:  '[' CHAR*? '/O]' -> skip;
OTHERS: CHAR+? -> skip;

CHAR:   ~[[\r\n]; 
//SEP:    '_'+? -> skip;
NEWLINE:'\r'?'\n' -> skip;
