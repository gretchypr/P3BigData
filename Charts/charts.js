$(document).ready(function(){
      displayCharts();
});
function displayCharts(val) {
     d3.csv("graph_data.csv", function(error, data) {
        if (error) throw error;

        new Chart(document.getElementById("flu_pie1"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[0]['p1']/data[0]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[0]['p2']/data[0]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[0]['p1'], data[0]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Flu from model 1',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 20,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("flu_pie2"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[1]['p1']/data[1]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[1]['p2']/data[1]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[1]['p1'], data[1]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Flu from model 2',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("zika_pie1"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[2]['p1']/data[2]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[2]['p2']/data[2]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[2]['p1'], data[2]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Zika from model 1',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("zika_pie2"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[3]['p1']/data[3]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[3]['p2']/data[3]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[3]['p1'], data[3]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Zika from model 2',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("dia_pie1"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[4]['p1']/data[4]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[4]['p2']/data[4]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[4]['p1'], data[4]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Diarrhea from model 1',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("dia_pie2"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[5]['p1']/data[5]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[5]['p2']/data[5]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[5]['p1'], data[5]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Diarrhea from model 2',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("ebo_pie1"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[6]['p1']/data[6]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[6]['p2']/data[6]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[6]['p1'], data[6]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Ebola from model 1',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("ebo_pie2"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[7]['p1']/data[7]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[7]['p2']/data[7]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[7]['p1'], data[7]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Ebola from model 2',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("mea_pie1"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[8]['p1']/data[8]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[8]['p2']/data[8]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[8]['p1'], data[8]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Measles from model 1',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
        new Chart(document.getElementById("mea_pie2"), {
            type: 'doughnut',
            data: {
              labels: ["Responses related to medical condition ("+((data[9]['p1']/data[9]['total'])*100).toFixed(2)+"%)",
              "Responses not related to medical condition ("+((data[9]['p2']/data[9]['total'])*100).toFixed(2)+"%)"],
              datasets: [{
                label: "Responses",
                backgroundColor: ["#58D68D", "#C0392B"],
                data: [data[9]['p1'], data[9]['p2']]
              }]
            },
            options: {
              responsive: false,
              maintainAspectRatio: true,
              title: {
                display: true,
                text: 'Responses of tweets related to Measles from model 2',
                fontSize: 25
              },
              legend: {
                labels: {
                  fontSize: 16
                }
              },
              tooltips: {
                bodyFontSize: 16,
                displayColors: false
              }
            }
        });
      });
}