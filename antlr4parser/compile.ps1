cd $PSScriptRoot

# ===== compile
# antlr4py3 .\RuleCheckTree.g4 -visitor
# antlr4 .\RuleCheckTree.g4
# javac RuleCheck*.java

# write "Successfully compiled."


# ===== grun quick test
cat input.log -First 1 | grun RuleCheckTree rctree -tokens -gui -tree