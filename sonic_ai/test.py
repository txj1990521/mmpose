from sonic_ai.Labelme2cocoKeypoints import labelme2coco_process

className = 'c'
projectName = 'CYS.220317-雅康-欣旺达切叠一体机'
inputImagePath = '/data2/5-标注数据/CYS.220317-雅康-欣旺达切叠一体机-定位/实验2-关键点/'
outputImagePath = 'D:/CYS.220317-雅康-欣旺达切叠一体机/result'
pointClassNum = 2
totalClassName = ["a", "b"]

dictConfig = dict(
    class_name=className,
    input=inputImagePath,
    output=outputImagePath,
    join_num=pointClassNum,
    project_name=projectName,
    total_classname=totalClassName,
    ratio=0.12,
)

labelme2coco_process(dictConfig)
