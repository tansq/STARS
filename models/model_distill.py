from .layers import *



class SRMSep_stem(nn.Module):

    def __init__(self, img_chs=3, affine=True):
        super(SRMSep_stem, self).__init__()

        self.affine = affine
        self.img_chs = img_chs

        self.first_stem = ConvLayer(1, 30, kernel_size=5, stride=1,
                                    affine=affine, act_func='relu', bias=True)

        self.first_stem.conv.weight.data = torch.from_numpy(SRM_npy).data
        print(self.first_stem.conv.weight.data)

    def forward(self, inputs):

        output_c1 = inputs[:, 0, :, :]
        output_c2 = inputs[:, 1, :, :]
        output_c3 = inputs[:, 2, :, :]
        out_c1 = output_c1.unsqueeze(1)
        out_c2 = output_c2.unsqueeze(1)
        out_c3 = output_c3.unsqueeze(1)
        c1 = self.first_stem(out_c1)
        c2 = self.first_stem(out_c2)
        c3 = self.first_stem(out_c3)
        out = torch.cat([c1, c2, c3], dim=1)  # 3*30=90

        return out
