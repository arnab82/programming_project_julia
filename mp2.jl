include("hf_new.jl") 
np=pyimport("numpy")
new_eri=zeros(Float64,nbasis,nbasis,nbasis,nbasis)
temp1=np.einsum("ip,ijkl->pjkl", c, eri)
temp2=np.einsum("jq,pjkl->pqkl", c, temp1)
temp3=np.einsum("kr,pqkl->pqrl", c, temp2)
new_eri=np.einsum("ls,pqrl->pqrs", c, temp3)
#println(new_eri)
println(size(new_eri))
E=np.array(e)
print(E)
emp2=[]
ndocc=5
hdfg,l_c=size(c)
println(l_c)
#=for p in 1:nbasis
    for q in 1:nbasis
        for r in 1:nbasis
            for s in 1:nbasis
                val=[]
                for i in 1:l_c
                    cip=c[i][p]
                    for j in 1:l_c
                        cjq=c[j][q]
                        for k in 1:l_c
                            ckr=c[k][r]
                            for l in 1:l_c
                                cls=c[l][s]
                                #newjk[p][q][r][s]+=C[i][p]*C[j][q]* \
                                # jk[i][j][k][l]*C[k][r]*C[l][s]
                                ertj=cip*cjq*eri[i][j][k][l]*ckr*cls
                                push!(val,ertj)
                            end
                        end
                    end
                end
                vahd=new_eri[p][q][r][s]
                push!(val,vahd)
                valu=sum(val)
            end
        end
    end
end=#
for i in 1:ndocc
    for a in (ndocc+1):nbasis
        for j in 1:ndocc
            for b in (ndocc+1):nbasis
                println(new_eri[i][a][j][b])
                x=(new_eri[i][a][j][b]*(2*new_eri[i][a][j][b]-new_eri[i][b][j][a]))/(E[i]+E[j]-E[a]-E[b])
                push!(emp2,x)
            end
        end
    end
end

print("MP2 correlation E = ",sum(emp2)) 

