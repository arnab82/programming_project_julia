using Einsum
using PyCall
np=pyimport("numpy")
include("hf_new.jl") 
include("mp2.jl") 
nofe=n_elec
nocc=Int64((nofe//2)+(nofe%2))
function mo_to_aso(new_eri)
    h,nbasis=size(new_eri)
    ASObasis=zeros(Float64,2*nbasis,2*nbasis,2*nbasis,2*nbasis)
    for i in 1:(2*nbasis)
        for j in 1:(2*nbasis)
            for k in 1:(2*nbasis)
                for l in 1:(2*nbasis)
                    ASObasis[i,j,k,l]=((new_eri[floor(Int64,i/2+0.5),floor(Int64,k/2+0.5),floor(Int64,j/2+0.5),floor(Int64,l/2+0.5)])*(i%2==k%2)*(j%2==l%2)-(new_eri[floor(Int64,i/2+0.5),floor(Int64,l/2+0.5),floor(Int64,j/2+0.5),floor(Int64,k/2+0.5)])*(i%2==l%2)*(j%2==k%2))
                end
            end
        end
    end
    return ASObasis
end
ASObasis=mo_to_aso(new_eri)
println("the value of ASObasis is",ASObasis)#SIGN PROBLEM
println(size(ASObasis))
exit()
function mo_to_cso(new_eri)
    h,nbasis=size(new_eri)
    CSObasis=zeros(Float64,2*nbasis,2*nbasis,2*nbasis,2*nbasis)
    for i in 1:(2*nbasis)
        for j in 1:(2*nbasis)
            for k in 1:(2*nbasis)
                for l in 1:(2*nbasis)
                    CSObasis[i,j,k,l]=(new_eri[floor(Int64,i/2+0.5),floor(Int64,k/2+0.5),floor(Int64,j/2+0.5),floor(Int64,l/2+0.5)])*(i%2==k%2)*(j%2==l%2) 
                end
            end
        end
    end
    return CSObasis
end
CSObasis=mo_to_cso(new_eri)
#println("the value of CSObasis is",CSObasis)
function Mat_aotoMat_mo(c,eri)
    h,nbasis=size(eri)
    temp1=zeros(Float64,nbasis,nbasis)
    Mat_mo=zeros(Float64,nbasis,nbasis)
    @einsum temp1[p,j]=*(c[i,p],eri[i,j])
    @einsum Mat_mo[p,q]=*(c[j,q],temp1[p,j])
    return Mat_mo
end
println("the value of mat_ao_to mat_mo is",Mat_aotoMat_mo(c,fock)[2,2])
println("the size of Mat_mo is",size(Mat_aotoMat_mo(c,fock)))
function Mat_motoMat_so(Mat_mo)
    h,nbasis=size(Mat_mo)
    Mat_so=zeros(Float64,2*nbasis,2*nbasis)
    for i in 1:(2*nbasis)
        Mat_so[i,i]=Mat_mo[floor(Int64,i/2+0.5),floor(Int64,i/2+0.5)]
    end
    return Mat_so
end
F_so=zeros(Float64,2*nbasis,2*nbasis)
F_so=Mat_motoMat_so(Mat_aotoMat_mo(c,fock))
#println("the value of fock matrix is ", F_so)#matched
#println(size(F_so))
o=nofe
#println(o)
v=2*nbasis-nofe
#println(v)
maxiter=500
ASObasis_oooo=zeros(Float64,o,o,o,o)
ASObasis_oovv=zeros(Float64,o,o,v,v)
ASObasis_ooov=zeros(Float64,o,o,o,v)
ASObasis_oovo=zeros(Float64,o,o,v,o)
ASObasis_vvoo=zeros(Float64,v,v,o,o)
ASObasis_vvvv=zeros(Float64,v,v,v,v)
ASObasis_vvov=zeros(Float64,v,v,o,v)
ASObasis_vvvo=zeros(Float64,v,v,v,o)
ASObasis_ovoo=zeros(Float64,o,v,o,o)
ASObasis_ovvv=zeros(Float64,o,v,v,v)
ASObasis_ovov=zeros(Float64,o,v,o,v)
ASObasis_ovvo=zeros(Float64,o,v,v,o)
ASObasis_vooo=zeros(Float64,v,o,o,o)
ASObasis_vovv=zeros(Float64,v,o,v,v)
ASObasis_voov=zeros(Float64,v,o,o,v)
ASObasis_vovo=zeros(Float64,v,o,v,o)
#For the partitioning of F_so matrix
function make_F_so_oo()
    F_so_oo=zeros(Float64,o,o)
    for i in 1:o
        for j in 1:o
            F_so_oo[i,j]=F_so[i,j]
        end
    end
    return F_so_oo
end

function make_F_so_vv()
    F_so_vv=zeros(Float64,v,v)
    for a in 1:v
        for b in 1:v
            F_so_vv[a,b]=F_so[a+nofe,b+nofe]
        end
    end
    return F_so_vv
end
#println("the value of vv block of fock is",make_F_so_vv())
#println("the value of oo block of fock is",make_F_so_oo())
function make_F_so_ov()
    F_so_ov=zeros(Float64,o,v)
    for i in 1:o
        for a in 1:v
            F_so_ov[i,a]=F_so[i,a+nofe]
        end
    end
    return F_so_ov
end
#println("the value of ov block of fock is",make_F_so_ov())
#println(size(make_F_so_oo()))
#println(size(make_F_so_ov()))
#println(size(make_F_so_vv()))#matched the oo,vv,ov,vo block fock
function make_F_so_vo()
    F_so_vo=zeros(Float64,v,o)
    for b in 1:v
        for j in 1:o
            F_so_vo[b,j]=F_so[b+nofe,j]
        end
    end
    return F_so_vo
end
#For the partitioning of ASObasis matrix
for i in 1:o
    for j in 1:o
        for m in 1:o
            for n in 1:o
                ASObasis_oooo[i,j,m,n]=ASObasis[i,j,m,n]
            end
        end
    end
end

for i in 1:o
    for j in 1:o
        for m in 1:o
            for b in 1:v
                ASObasis_ooov[i,j,m,b]=ASObasis[i,j,m,b+nofe]
            end
        end
    end
end

for i in 1:o
    for j in 1:o
        for a in 1:v
            for m in 1:o
                ASObasis_oovo[i,j,a,m]=ASObasis[i,j,a+nofe,m]
            end
        end
    end
end

for i in 1:o
    for a in 1:v
        for j in 1:o
            for m in 1:o
                ASObasis_ovoo[i,a,j,m]=ASObasis[i,a+nofe,j,m]
            end
        end
    end
end

for a in 1:v
    for j in 1:o    
        for m in 1:o
            for n in 1:o
                ASObasis_vooo[a,j,m,n]=ASObasis[a+nofe,j,m,n]
            end
        end
    end
end

for a in 1:v
    for b in 1:v
        for m in 1:o
            for n in 1:o
                ASObasis_vvoo[a,b,m,n]=ASObasis[a+nofe,b+nofe,m,n]
            end
        end
    end
end

for i in 1:o
    for a in 1:v
        for b in 1:v
            for n in 1:o
                ASObasis_ovvo[i,a,b,n]=ASObasis[i,a+nofe,b+nofe,n]
            end
        end
    end
end

for i in 1:o
    for a in 1:v
        for m in 1:o
            for b in 1:v
                ASObasis_ovov[i,a,m,b]=ASObasis[i,a+nofe,m,b+nofe]
            end
        end
    end
end

for i in 1:o
    for j in 1:o
        for a in 1:v
            for b in 1:v
                ASObasis_oovv[i,j,a,b]=ASObasis[i,j,a+nofe,b+nofe]
            end
        end
    end
end
#println("the value of double excitation operator is",ASObasis_oovv,"\n\n")
for a in 1:v
    for b in 1:v
        for e in 1:v
            for f in 1:v
                ASObasis_vvvv[a,b,e,f]=ASObasis[a+nofe,b+nofe,e+nofe,f+nofe]
            end
        end
    end
end

for a in 1:v
    for b in 1:v
        for m in 1:o
            for f in 1:v
                ASObasis_vvov[a,b,m,f]=ASObasis[a+nofe,b+nofe,m,f+nofe]
            end
        end
    end
end

for a in 1:v
    for b in 1:v
        for e in 1:v
            for n in 1:o
                ASObasis_vvvo[a,b,e,n]=ASObasis[a+nofe,b+nofe,e+nofe,n]
            end
        end
    end
end

for i in 1:o
    for b in 1:v
        for e in 1:v
            for f in 1:v
                ASObasis_ovvv[i,b,e,f]=ASObasis[i,b+nofe,e+nofe,f+nofe]
            end
        end
    end
end

for a in 1:v
    for j in 1:o
        for e in 1:v
            for f in 1:v
                ASObasis_vovv[a,j,e,f]=ASObasis[a+nofe,j,e+nofe,f+nofe]
            end
        end
    end
end

for a in 1:v
    for j in 1:o
        for m in 1:o
            for f in 1:v
                ASObasis_voov[a,j,m,f]=ASObasis[a+nofe,j,m,f+nofe]
            end
        end
    end
end
for a in 1:v
    for j in 1:o
        for e in 1:v
            for n in 1:o
                ASObasis_vovo[a,j,e,n]=ASObasis[a+nofe,j,e+nofe,n]
            end
        end
    end
end
function make_td()
    td=zeros(Float64,o,o,v,v)
    for i in 1:o
        for j in 1:o
            for a in 1:v
                for b in 1:v
                    td[i,j,a,b]=(ASObasis_oovv[i,j,a,b])/(E[floor(Int64,i/2+0.5)]+E[floor(Int64,j/2+0.5)]-E[floor(Int64,(a+nofe)/2+0.5)]-E[floor(Int64,(b+nofe)/2+0.5)])
                end
            end
        end
    end
    return td
end
#println("the value of double excitation operator is ",make_td(),"\n\n")
function mp2_so(ASObasis_oovv,td)
    E_mp2_so=0.0
    for i in 1:o
        for j in 1:o
            for a in 1:v
                for b in 1:v
                    E_mp2_so+=0.25*(ASObasis_oovv[i,j,a,b])*td[i,j,a,b]
                end
            end
        end
    end
    return E_mp2_so
end
println("the value of mp2 correlation energy in so basis is ",mp2_so(ASObasis_oovv,make_td()))
#Step #3: Calculate the CC Intermediates
function cind_r1r2(p,q,r1,r2,nofe)
    return (r2*(p-nofe*(p>=nofe)))+q-nofe*(q>=nofe)
end
function denomin(F_so_oo,F_so_vv)
    D=[]
    for i in 1:o
        for a in 1:v
            x=F_so_oo[i,i]-F_so_vv[a,a]
            push!(D,x)
        end
    end
    return D
end
println("the value of initial denominator is",denomin(make_F_so_oo(),make_F_so_vv()))
#iteration
function ccsd_scf()
    E_ccsd=0.0
    td=zeros(Float64,o,o,v,v)
    td=make_td()
    tao=zeros(Float64,o,o,v,v)
    ts=zeros(Float64,o,v)
    tsnew=zeros(Float64,o,v)
    tdnew=zeros(Float64,o,o,v,v)
    Fae=zeros(Float64,v,v)
    Fmi=zeros(Float64,o,o)
    Fme=zeros(Float64,o,v)
    Wmnij=zeros(Float64,o,o,o,o)
    Wabef=zeros(Float64,v,v,v,v)
    Wmbej=zeros(Float64,o,v,v,o)
    T1=zeros(Float64,o,v)
    taobar=zeros(Float64,o,o,v,v)
    F_so_vv=zeros(Float64,v,v)
    F_so_ov=zeros(Float64,o,v)
    F_so_oo=zeros(Float64,o,o)
    F_so_ov=make_F_so_ov()
    F_so_vv=make_F_so_vv()
    F_so_oo=make_F_so_oo()
    Wmnij_a=zeros(Float64,o,o,o,o)
    Wmnij_b=zeros(Float64,o,o,o,o)
    Wabef_a=zeros(Float64,v,v,v,v)
    Wabef_b=zeros(Float64,v,v,v,v)
    Wmbej_a=zeros(Float64,o,v,v,o)
    Wmbej_b=zeros(Float64,o,v,v,o)
    Wmbej_c=zeros(Float64,o,v,v,o)
    T2_a=zeros(Float64,o,o,v,v)
    T2_b=zeros(Float64,o,o,v,v)
    T2_c=zeros(Float64,o,o,v,v)
    T2_d=zeros(Float64,o,o,v,v)
    T2_e=zeros(Float64,o,o,v,v)
    temp_T2_e=zeros(Float64,o,o,v,v)
    T2_f=zeros(Float64,o,o,v,v)
    T2_g=zeros(Float64,o,o,v,v)
    temp_be=zeros(Float64,v,v)
    temp_jm=zeros(Float64,o,o)
    D=denomin(F_so_oo,F_so_vv)
    E_ccsd_a=0.0
    E_ccsd_b=0.0
    E_ccsd_c=0.0
    #for iteration in 1:maxiter
        #formation of tao
    for j in 1:o
        for i in 1:o
            for a in 1:v
                for b in 1:v
                    tao[i,j,a,b]=td[i,j,a,b] + (ts[i,a]*ts[j,b]-ts[i,b]*ts[j,a])
                end
            end
        end
    end
    println("the value of tao is",tao)
    println("the value ofsize of tao is",size(tao))
        #formation of taobar
    for i in 1:o
        for j in 1:o
            for a in 1:v
                for b in 1:v
                    taobar[i,j,a,b]=td[i,j,a,b] +0.5*(ts[i,a]*ts[j,b]-ts[i,b]*ts[j,a])
                end
            end
        end
    end
    println("the value of taobar is",taobar)
    #intermediates
    # for Fae
    @einsum Fae[a,e]= ts[m,e]*F_so_ov[m,a]
    @einsum Fae[a,e]=(-0.5*Fae[a,e])+(ts[m,f]*ASObasis_ovvv[m,a,f,e])
    @einsum Fae[a,e]=Fae[a,e]-0.5*(taobar[m,n,a,f]*ASObasis_oovv[m,n,e,f])
    #for Fmi
    @einsum Fmi[m,i]=ts[i,e]*F_so_ov[m,e]
    @einsum Fmi[m,i]=(0.5*Fmi[m,i])+(ts[n,e]*ASObasis_ooov[m,n,i,e])
    @einsum Fmi[m,i]=Fmi[m,i]+0.5*(taobar[i,n,e,f]*ASObasis_oovv[m,n,e,f])
    #for Fme
    @einsum Fme[m,e]=ts[n,f]*ASObasis_oovv[m,n,e,f]
    for a in 1:v
        for e in 1:v
            Fae[a,e]=(1-(a==e))*F_so_vv[a,e] +Fae[a,e]
        end
    end
    for m in 1:o
        for i in 1:o
            Fmi[m,i] = (1-(m==i))*F_so_oo[m,i] + Fmi[m,i]
        end
    end
    println("the fmi matrix is",Fmi)
    println("the fae matrix is",Fae)
    for m in 1:o
        for e in 1:v
            Fme[m,e] = F_so_ov[m,e] + Fme[m,e]
        end
    end
    println("the fme matrix is",Fme)
    #W intermediates       
    @einsum Wmnij_a[m,n,i,j]=ts[j,e]*ASObasis_ooov[m,n,i,e]
    @einsum Wmnij_b[m,n,i,j]=tao[i,j,e,f]*ASObasis_oovv[m,n,e,f]
    @einsum Wabef_a[a,b,e,f]=ts[m,b]*ASObasis_vovv[a,m,e,f]
    @einsum Wabef_b[a,b,e,f]=tao[m,n,a,b]*ASObasis_oovv[m,n,e,f]
    temp_w=zeros(Float64,o,o,v,v)
    @einsum Wmbej_a[m,b,e,j]=ts[j,f]*ASObasis_ovvv[m,b,e,f]
    @einsum Wmbej_b[m,b,e,j]=ts[n,b]*ASObasis_oovo[m,n,e,j]
    #Wmbej_c=np.einsum("jnfb,mnef->mbej",(0.5*td+np.einsum("jf,nb->jnfb",ts,ts)),ASObasis_oovv)
    @einsum temp_w[j,n,f,b]=ts[j,f]*ts[n,b]
    temp_w=(0.5*td+temp_w)
    @einsum Wmbej_c[m,b,e,j]=temp_w[j,n,f,b]*ASObasis_oovv[m,n,e,f]
    #evaluating W intermediates
    for m in 1:o
        for n in 1:o
            for i in 1:o
                for j in 1:o
                    Wmnij[m,n,i,j]=ASObasis_oooo[m,n,i,j]+ Wmnij_a[m,n,i,j] - Wmnij_a[m,n,j,i] + 0.25*Wmnij_b[m,n,i,j]
                end
            end
        end
    end

    for a in 1:v
        for b in 1:v
            for e in 1:v
                for f in 1:v
                    Wabef[a,b,e,f]=ASObasis_vvvv[a,b,e,f]-Wabef_a[a,b,e,f]+Wabef_a[b,a,e,f] +0.25*Wabef_b[a,b,e,f]
                end
            end
        end
    end
    for m in 1:o
        for b in 1:v
            for e in 1:v
                for j in 1:o
                    Wmbej[m,b,e,j]=ASObasis_ovvo[m,b,e,j]+ Wmbej_a[m,b,e,j] - Wmbej_b[m,b,e,j] - Wmbej_c[m,b,e,j]
                end
            end
        end
    end
#For T1
    @einsum T1[i,a]=ts[i,e]*Fae[a,e]
    @einsum T1[i,a]=T1[i,a]-(ts[m,a]*Fmi[m,i])
    @einsum T1[i,a]=T1[i,a]+(td[i,m,a,e]*Fme[m,e])
    @einsum T1[i,a]=T1[i,a]-(ts[n,f]*ASObasis_ovov[n,a,i,f])
    @einsum T1[i,a]=T1[i,a]-0.5*(td[i,m,e,f]*ASObasis_ovvv[m,a,e,f])
    @einsum T1[i,a]=T1[i,a]-0.5*(td[m,n,a,e]*ASObasis_oovo[n,m,e,i])
    println("the value of T1_f is",T1,"\n\n\n")
#for T2 
    @einsum temp_be[b,e]=ts[m,b]*Fme[m,e]
    temp_be=Fae-0.5*temp_be
    @einsum T2_a[i,j,a,b]=td[i,j,a,e]*temp_be[b,e]
    @einsum temp_jm[m,j]=ts[j,e]*Fme[m,e]
    temp_jm=Fmi-0.5*temp_jm
    @einsum T2_b[i,j,a,b]=td[i,m,a,b]*temp_jm[m,j]
    @einsum T2_c[i,j,a,b]=tao[m,n,a,b]*Wmnij[m,n,i,j]
    @einsum T2_d[i,j,a,b]=tao[i,j,e,f]*Wabef[a,b,e,f]
    @einsum T2_e[i,j,a,b]=td[i,m,a,e]*Wmbej[m,b,e,j]
    @einsum temp_T2_e[i,j,a,b]=ts[i,e]*ts[m,a]*ASObasis_ovvo[m,b,e,j]
    T2_e=T2_e-temp_T2_e
    @einsum T2_f[i,j,a,b]=ts[i,e]*ASObasis_vvvo[a,b,e,j]
    @einsum T2_g[i,j,a,b]=ts[m,a]*ASObasis_ovoo[m,b,i,j]
    #println("the value of T2_g is",T2_g)
#Writing T1 Equation
    for i in 1:o
        for a in 1:v
            tsnew[i,a] = (1/D[i*a]) * (F_so_ov[i,a] + T1[i,a])
        end
    end
    println("the matrix tsnew is",tsnew,"\n\n")
#writing T2 Equation
    for i in 1:o
        for j in 1:o
            for a in 1:v
                for b in 1:v
                    tdnew[i,j,a,b]=(1/(D[i*a]+D[j*b]))*(ASObasis_oovv[i,j,a,b]+ T2_a[i,j,a,b] - T2_a[i,j,b,a] - T2_b[i,j,a,b] + T2_b[j,i,a,b] + 0.5*T2_c[i,j,a,b]+ 0.5*T2_d[i,j,a,b] + T2_e[i,j,a,b] - T2_e[j,i,a,b] - T2_e[i,j,b,a] + T2_e[j,i,b,a]+ T2_f[i,j,a,b] - T2_f[j,i,a,b] - T2_g[i,j,a,b] + T2_g[i,j,b,a])
                end
            end
    end
    end
    #println("the matrix tdnew is",tdnew,"\n\n")    #E_ccsd calculation     
    for i in 1:o
        for a in 1:v
            E_ccsd_a+=F_so_ov[i,a]*tsnew[i,a]
        end
    end
    for i in 1:o
        for j in 1:o
            for a in 1:v
                for b in 1:v
                    E_ccsd_b+=ASObasis_oovv[i,j,a,b]*tdnew[i,j,a,b]
                    E_ccsd_c+=ASObasis_oovv[i,j,a,b]*tsnew[i,a]*tsnew[j,b]
                end
            end
        end
    end
    println("the value of E_ccsd_a is", E_ccsd_a)
    println("the value of E_ccsd_c is", E_ccsd_c)
    println("the value of E_ccsd_b is", E_ccsd_b)
    E_ccsdnew=E_ccsd_a+(0.25*E_ccsd_b)+(0.5*E_ccsd_c)
    #Check for convergence
    println("iteration= ","1","    energy= ",E_ccsdnew,"Delta_e",E_ccsdnew-E_ccsd)  
        #=if abs(E_ccsdnew-E_ccsd)<=10^(-10) && abs(np.std(tsnew-ts))<=10^(-10) && abs(np.std(tdnew-td))<=10^(-10)
            println("SUCCESS! Coupled Cluster SCF converged.")
            ts=tsnew
            td=tdnew
            E_ccsd=E_ccsdnew
            print("E_ccsd = " + str(E_ccsd) + ".")
            break
        end
        if iteration==maxiter-1
            println("Maximum iterations reached. CCSD SCF did not converge.")
            exit()
        end
        Delta_E_ccsd=E_ccsdnew-E_ccsd
        Delta_ts=np.std(tsnew-ts)
        Delta_td=np.std(tdnew-td)
        if iteration==0
            println("\n\n\n")
            println("\n\n\n\nCCSD SCF Iterations:\n--------------------")
            println("\n\n\tIteration(s)\t\tE_ccsd(new)\t\tDelta_E_ccsd\t\tDelta_ts\t\tDelta_td\n")
        println("\n\t'+str(iteration+1)+'\t\t'+str(E_ccsdnew)+'\t'+str(Delta_E_ccsd)+'\t'+str(Delta_ts)+'\t\t'+str(Delta_td)+'\n")

        println("CCSD SCF energy deviations=" + str(E_ccsdnew-E_ccsd) + ", singles=" + str(abs(np.std(tsnew-ts))) + " and doubles=" + str(abs(np.std(tdnew-td))) + " in " + str(iteration+1) + " iteration(s).")
        end
        E_ccsd=E_ccsdnew
        ts=tsnew
        td=tdnew
    end=#
    return E_ccsdnew,td,ts,Fme,Fae,Fmi,tao,taobar#,Delta_E_ccsd,Delta_ts,Delta_td
end
println("The value of E_ccsd is",ccsd_scf())