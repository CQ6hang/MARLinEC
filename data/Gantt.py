import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import datetime


class Plotter:
    def __init__(self, n_wf, n_vm, vm_name, result):

        # plotly authentication
        plotly.tools.set_credentials_file(username='HAnGkNew', api_key='oU6TzXzVVvFwz0HPkCyf')

        self.n_wf = n_wf
        self.n_vm = n_vm
        self.vm_name = vm_name
        self.result = result

    def plot(self):
        df = []
        result_format = []
        for i in self.result:
            start_time = str(datetime.timedelta(seconds=i[2]))
            end_time = str(datetime.timedelta(seconds=i[3]))
            result_format.append((i[0], i[1], [start_time, end_time]))

        for i in range(self.n_vm):
            for j in range(self.n_wf):
                for k in result_format:
                    if (i, j) == (k[1], k[0]):
                        df.append(
                            dict(Task=self.vm_name[i], Start='2019-05-13 ' + k[2][0], Finish='2019-05-13 ' + k[2][1],
                                 Resource='Workflow %s' % (j + 1)))

        fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True,
                              title='Workflow Schedule')
        py.plot(fig, filename='DQN_workflow_scheduling', world_readable=True)
