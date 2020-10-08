[CmdletBinding()]
Param(
    [switch]$Remove
)

$pwd0=$pwd
cd $PSScriptRoot # logs

'bak','bak/csv','bak/runs' | %{ if(!(Test-Path $_)){ md $_} }

if($Remove){
    ri train.log -ea si
    ri csv/*
    ri -r runs/*
}else{
    "`n`n`n########################################`n`n`n" >> bak/train.log
    cat train.log >> bak/train.log
    '' > train.log
    mi runs/* bak/runs
    mi csv/* bak/csv
}

cd $pwd0
