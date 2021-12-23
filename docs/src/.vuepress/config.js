const { description } = require('../../package')

module.exports = {
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: "",
  publicPath: './',

  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#description
   */
  description: description,

  /**
   * Extra tags to be injected to the page HTML `<head>`
   *
   * ref：https://v1.vuepress.vuejs.org/config/#head
   */
  head: [
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],

  /**
   * Theme configuration, here is the default theme configuration for VuePress.
   *
   * ref：https://v1.vuepress.vuejs.org/theme/default-theme-config.html
   */
  themeConfig: {
    logo: 'damavand.png',
    repo: '',
    editLinks: false,
    docsDir: '',
    editLinkText: '',
    lastUpdated: false,
    displayAllHeaders: true,
    nav: [
      {
        text: 'Install',
        link: '/install/'
      },
      {
        text: 'Guide',
        link: '/guide/',
      },
      {
        text: 'HPC',
        link: '/hpc/',
      },
      {
        text: 'PennyLane',
        link: '/pennylane-plugin/',
      },
      {
        text: 'prevision-quantum-nn',
        link: '/prevision-quantum-nn/',
      },
      {
        text: 'Github',
        link: 'https://github.com/previsionio/damavand'
      }
    ],
    sidebar: ['/', '/install/', '/guide/', '/hpc/', '/pennylane-plugin/', '/prevision-quantum-nn/']
  },

  /**
   * Apply plugins，ref：https://v1.vuepress.vuejs.org/zh/plugin/
   */
  plugins: [
    '@vuepress/plugin-back-to-top',
    '@vuepress/plugin-medium-zoom',
  ]
}
