    def get_TDS_constant_1trap(dE,t,T):
        k=0.025/300
        f=np.exp(dE/k/T)*t
        c=1-np.exp(-f)
        return c*10**6,f

    def get_TDS_ramp_1trap(dE,T0,T1,beta):
        k=0.025/300
        Ei0=sc.expi(dE/k/T0)
        Ei1=sc.expi(dE/k/T1)
        f0=(T0*np.exp(dE/k/T0)-dE*Ei0/k)/beta
        f1=(T1*np.exp(dE/k/T1)-dE*Ei1/k)/beta
        c=1-np.exp(-(f1-f0))
        return c*10**6,f0,f1

    def get_TDS_rate_ramp_1trap(dE,T0,T1,beta):
        k=0.025/300
        Ei1=sc.expi(dE/k/T1)
        dc=np.exp((dE*beta+dE*T1*Ei1+k*T1**2*(-np.exp(dE/k/T1)))/(beta*k*T1))/beta
        return dc*10**6

    def get_integral_from_const(CC,N,dT,Tstart,Tfinal):
        NN=np.arange(Tstart,Tfinal,dT)
        for i in range(1,len(CC)+1):
            NN[i-1]=N
        for i in range(1,len(CC)+1):
            if i==1:
                NN[i-1]=N-CC[i-1]*N
            NN[i-1]=NN[i-2]-CC[i-1]*NN[i-2]
        for i in range(1,len(CC)+1):
            NN[i-1]=N-NN[i-1]
        dN=np.diff(NN)/dT
        dN1 = np.append(0,dN)
        return NN/N,(10**6)*dN1/N

    def get_ramp_from_const(dT,dTdt,Tstart,Tfinal,dE):
        dt= dT/dTdt
        TT=np.arange(Tstart,Tfinal,dT)
        CC=np.arange(Tstart,Tfinal,dT)
        for i in range(1,len(CC)+1):
            Tcon=TT[i-1]
            CCC,ff=get_TDS_constant_1trap(dE,dt,Tcon)
            CC[i-1]=CCC#*10**(-6)
        return CC*10**(-6)
