grammar RuleCheckTree;

/* Parser Rules */
//prs: may not complete: ((())], (((())]], [(()))
//pr:  complete: ((()()))()
rctree: prs+ NEWLINE*?;

prs:    pr
    |   PROP+ pr
    |   pr req
    |   req
;

pr:     PROP pr+ req
    |   PROP req
;

req: CMP? ROBJ? RPROP;


/* Lexer Rules*/
//non-greedy
PROP:   '[' CHAR*? '/prop' 'x'? ']';
CMP:    '[' CHAR*? '/cmp]';
ROBJ:   '[' CHAR*? '/' 'a'? 'Robj]';
RPROP:  '[' CHAR*? '/' 'a'? 'Rprop' 'x'? ']';
OBJ:    '[' CHAR*? '/obj]' ->skip;
OTHER:  '[' CHAR*? '/O]' -> skip;
OTHERS: CHAR+? -> skip;

CHAR:   ~[_\r\n];
SEP:    '_'+? -> skip;
NEWLINE:'\r'?'\n';
