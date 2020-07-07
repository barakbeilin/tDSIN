class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj,self.data = (img1,img2),[-1+2*img1.data,-1+2*img2.data]

https://docs.fast.ai/tutorial.itemlist.html#Example:-ImageTupleList



 # learner - how to creat learner then save model + callback for learning-rate scheduling and weight-decay.
https://docs.fast.ai/basic_train.html#Learner