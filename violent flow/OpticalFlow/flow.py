import bob.ip.optflow.liu.sor

class OptFlow:
    def __init__(self):
        self.alpha = 0.0026
        self.ratio = 0.6
        self.minWidth = 20
        self.nOuterFPIterations = 7
        self.nInnerFPIterations = 1
        self.nSORIterations = 30
        self.flows = ()

    def sorFlow(self,frame1,frame2):
        #(vx,vy,w) = bob.ip.optflow.liu.sor.flow(frame1,frame2,self.alpha,self.ratio,self.minWidth,self.nOuterFPIterations,self.nInnerFPIterations,self.nSORIterations)
        self.flows = bob.ip.optflow.liu.sor.flow(frame1,frame2)
        return self.flows
