cd $PSScriptRoot

antlr4py3 .\RuleCheckTree.g4 -visitor
antlr4 .\RuleCheckTree.g4
javac RuleCheck*.java

write "Successfully compiled."
