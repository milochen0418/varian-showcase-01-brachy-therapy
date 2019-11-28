
The process will get AI output and then wrap it into RS file.  
The main process is in main.py.  

# main.py  
INPUT:  
The input of main function are CT folder and AI Model name. 
OUPUT: 
The output is RS file.   

DEPENDENCY:
The main.py use AI_process.py and SimpleInterpolateRsWrapup.py  

# AI_process.py  
INPUT: 
main.py will prepare AI model name and CT filepath list as input for  
the function AI_process() in AI_process.py.  

OUTPUT:  
The output will be AI output output result.  

DEPENDENCY:  
The AI_Process.py use Mult_Class_Brachy.py to process the case of AI Brachy.  
The AI_Process.py will use more and more new model in future.   

EXAMPLE:  
refer main.py will understand how to use AI_process.py

# Mult_Class_Brachy.py  
The Brachy AI code from another AI developer.      

# SimpleInterpolateRsWrapUp.py  
This python code will wrap AI output to RS file.   
INPUT:   
AI output  

OUTPUT:  
RS file   


