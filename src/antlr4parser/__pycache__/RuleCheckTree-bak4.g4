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


/* Lexer Rules */
// use non-greedy
PROP:   STRING '#prop' 'x'?;
CMP:    STRING '#cmp';
ROBJ:   STRING '#' 'a'? 'Robj';
RPROP:  STRING '#' 'a'? 'Rprop' 'x'?;
OBJ:    STRING '#obj' -> skip;
OTHER:  STRING '#O' -> skip;

fragment STRING: (~[# \t\r\n])*?; // .*?
NEWLINE: '\r'?'\n';
WS:     [ \t]+ -> skip;
COMMENT: '##' .*? NEWLINE -> skip ;


/* Example: 
aaa#prop bb#cmp cccc#Rprop*/
