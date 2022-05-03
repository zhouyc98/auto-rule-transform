grammar RuleCheckTree;

/* Parser Rules */
// prs: all p-r pattern: ((())], (((())]], [(()))
// pr:  complete prop-req pattern: ((()()))()
rctree: prs+;
prs:    pr
    |   PROP+ pr
    |   pr? req
;
pr:     PROP pr+ req
    |   PROP req
;
req: CMP? ROBJ? (RPROP|ARPROP);

/* Lexer Rules*/
//non-greedy
PROP:   '[' CHAR*? '/prop]';
CMP:    '[' CHAR*? '/cmp]';
ROBJ:   '[' CHAR*? '/Robj]';
RPROP:  '[' CHAR*? '/Rprop]';
ARPROP: '[' CHAR*? '/ARprop]';
OBJ:    '[' CHAR*? '/obj]' ->skip;
OTHER:  '[' CHAR*? '/O]' -> skip;
OTHERS: CHAR+? -> skip;
CHAR:   ~[[\r\n]; 
NEWLINE:'\r'?'\n' -> skip;
