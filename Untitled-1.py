lst=eval(input("enter the list:"))
start=eval(input("enter the start list:"))
end=eval(input("enter the end list:"))
def ite(lst,start,end):
    if start<0 or start>=end:
        return
    print(lst[start])
    ite(lst,start+1,end)
ite(lst,start,end)