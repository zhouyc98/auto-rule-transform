grammar RuleCheckTree;

/* Parser Rules */
rctree:   propreqs+ NEWLINE?;

propreqs:   props propreqs rprops
    |   props rprops
    |   rprops props
    |   rprops
;

props:  PROP;               //a single prop[x]
rprops: CMP? ROBJ? RPROP+;  //a single req prop[x]


/* Lexer Rules */
//use non-greedy
PROP:   '['.*? '/prop' 'x'? ']';
ROBJ:   '['.*? '/' 'a'? 'Robj]';
RPROP:  '['.*? '/' 'a'? 'Rprop' 'x'? ']';
CMP:    '['.*? '/cmp]';

NEWLINE:'\r'?'\n';
OBJ:    '['.*? '/obj]' -> skip;
WS:     [ \t]+ -> skip;
COMMENT: '#' .*? NEWLINE -> skip;
