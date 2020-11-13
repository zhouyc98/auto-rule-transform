grammar RuleCheckTree;

/* Parser Rules */
//prs: may not complete: ((())], (((())]], [(()))
//pr:  complete: ((()()))()
//RCTree
rctree: prs+ NEWLINE*?;

//not fully matched pr pair
prs:    pr
    |   PROP+ pr
    |   pr req
    |   req
;

//prop-req pair
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

CHAR:   ~[_\r\n]; 
SEP:    '_'+? -> skip;
NEWLINE:'\r'?'\n';
