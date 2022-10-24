include("rhf.jl") 
time=PyCall.pyimport("time")
np=PyCall.pyimport("numpy")
start=time.time()
using TensorOperations
using Einsum
new_eri=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
neweri=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
temp1=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
temp2=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
temp3=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
temp1=np.einsum("ip,ijkl->pjkl", c, eri)
temp2=np.einsum("jq,pjkl->pqkl", c, temp1)
temp3=np.einsum("kr,pqkl->pqrl", c, temp2)
neweri=np.einsum("ls,pqrl->pqrs", c, temp3)
@einsum temp1[p,j,k,l]=(c[i,p]*eri[i,j,k,l])
#@tensor temp1[p,j,k,l]=(c[i,p]*eri[i,j,k,l])
@einsum temp2[p,q,k,l]=(c[j,q]*temp1[p,j,k,l])
#@tensor temp2[p,q,k,l]=(c[j,q]*temp1[p,j,k,l])
@einsum temp3[p,q,r,l]=(c[k,r]*temp2[p,q,k,l])
#@tensor temp3[p,q,r,l]=(c[k,r]*temp2[p,q,k,l])
@einsum new_eri[p,q,r,s]=(c[l,s]*temp3[p,q,r,l])
#@tensor new_eri[p,q,r,s]=(c[l,s]*temp3[p,q,r,l])
#println("the type of new_eri is ",typeof(new_eri))
#println("THE NEW_ERI IS" ,new_eri,"\n\n\n\n")
#println("THE NEWERI IS" ,neweri,"\n\n\n\n")
#println("THE ERI IS" ,eri,"\n\n")
#println(size(new_eri),"\n")
ndocc=5
#=function noddy_algorithm()
    hdfg,l_c=size(c)
    println(l_c)
    for p in 1:nbasis
        for q in 1:nbasis
            for r in 1:nbasis
                for s in 1:nbasis
                    for i in 1:l_c
                        cip=c[i,p]
                        for j in 1:l_c
                            cjq=c[j,q]
                            for k in 1:l_c
                                ckr=c[k,r]
                                for l in 1:l_c
                                    cls=c[l,s]
                                #newjk[p][q][r][s]+=C[i][p]*C[j][q]* \
                                # jk[i][j][k][l]*C[k][r]*C[l][s]
                                new_eri[i,j,k,l]+=cip*cjq*eri[i,j,k,l]*ckr*cls
                                
                                end
                            end
                        end
                    end
                end
                return new_eri
            end
        end
    end
end
new_eri_=noddy_algorithm()
println("the value of new_eri is",new_eri,"\n")
println("the type of new_eri is ",typeof(new_eri))=#
function compute_mp2(new_eri)
    emp2=0.0
    for i in 1:ndocc
        for a in (ndocc+1):nbasis
            for j in 1:ndocc
                for b in (ndocc+1):nbasis
                    emp2+=((*(new_eri[i,a,j,b],(*(2,new_eri[i,a,j,b])-new_eri[i,b,j,a])))/(E[i]+E[j]-E[a]-E[b]))
                end
            end
        end
    end
    return emp2
end
print("MP2 correlation Energy = ",compute_mp2(new_eri),"\n")
print("Total Energy = ",compute_mp2(new_eri)+Hartree_fock_energy,"\n")
endtime=time.time()
println("the runtime of the code is", endtime-start)

