import streamlit as st


class AboutMe:

    @staticmethod
    def main():
        """
        ref: https://qiita.com/s-yoshiki/items/436bbe1f7160b610b05c
        :return:
        """

        with st.expander(label='About me', expanded=True):
            st.markdown('### About me')

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown('#### Link')
                st.markdown(
                    f'[![](https://img.shields.io/badge/-GitHub-181717.svg?logo=github&style=flat)](https://github.com/tn1994)'
                )
                st.markdown(
                    '[![](https://img.shields.io/badge/-zenn-232F3E.svg?logo=zenn&style=flat)](https://zenn.dev/tn1994)')
                st.markdown(
                    '[![](https://img.shields.io/badge/-kaggle-232F3E.svg?logo=kaggle&style=flat)](https://www.kaggle.com/takamichinosho)')
                st.markdown(
                    '[![](https://img.shields.io/badge/-Apple%20Music-232F3E.svg?logo=applemusic&style=flat)](https://music.apple.com/jp/playlist/ランダムリスト/pl.u-GgA5kl6cd2Bx7q)'
                )
                _twitter_user_name: str = 'tn_learninging'
                st.markdown(
                    f'[![](https://img.shields.io/badge/-Twitter-232F3E.svg?logo=twitter&style=flat)](https://twitter.com/{_twitter_user_name})')

            with c2:
                st.markdown('#### Code')
                get_badge_as_markdown(logo='Python')
                get_badge_as_markdown(logo='TypeScript')
                get_badge_as_markdown(logo='Docker')
                get_badge_as_markdown(logo='', subject='docker%20compose')
                get_badge_as_markdown(logo='Terraform')
                get_badge_as_markdown(logo='', subject='shell%20script')

            with c3:
                st.markdown('#### Framework & Library')
                get_badge_as_markdown(logo='Django')
                get_badge_as_markdown(logo='PyTorch')
                get_badge_as_markdown(logo='Streamlit')
                get_badge_as_markdown(logo='React')
                get_badge_as_markdown(logo='scikitlearn')
                get_badge_as_markdown(logo='Numpy')
                get_badge_as_markdown(logo='Pandas')
                get_badge_as_markdown(logo='SciPy')

            with c4:
                st.markdown('#### Infra')
                with st.container():
                    get_badge_as_markdown(subject='Amazon%20AWS', logo='amazon-aws')
                    get_badge_as_markdown(logo='Amazon%20EC2')
                    get_badge_as_markdown(logo='Amazon%20RDS')
                    get_badge_as_markdown(logo='Amazon%20S3')
                    get_badge_as_markdown(logo='Amazon%20DynamoDB')
                    get_badge_as_markdown(logo='Amazon%20CloudWatch')
                    get_badge_as_markdown(logo='AWS%20Lambda')
                    get_badge_as_markdown(logo='AWS%20Fargate')

            with c5:
                st.markdown('#### Infra')
                get_badge_as_markdown(subject='Amazon%20SageMaker', logo='amazon-aws')
                get_badge_as_markdown(subject='Amazon%20CodePipeline', logo='amazon-aws')

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown('#### Deploy')
                get_badge_as_markdown(logo='Linux')
                get_badge_as_markdown(logo='Ubuntu')
                get_badge_as_markdown(logo='RaspberryPi')
                get_badge_as_markdown(logo='NVIDIA')

            with c2:
                st.markdown('#### IDE')
                get_badge_as_markdown(logo='Pycharm')

            with c3:
                st.markdown('#### Other')
                get_badge_as_markdown(logo='macOS')
                get_badge_as_markdown(logo='Slack')
                get_badge_as_markdown(logo='Confluence')

            with c4:
                st.markdown('#### Like')
                get_badge_as_markdown(logo='HHKB')
                get_badge_as_markdown(logo='Sennheiser')
                get_badge_as_markdown(logo='', subject='K%20POP')
                get_badge_as_markdown(logo='', subject='R&B')

            st.markdown('### Github Stats')
            st.markdown(
                "![tn1994's github stats](https://github-readme-stats.vercel.app/api?username=tn1994&count_private=true&show_icons=true&theme=radical)")
            st.markdown(
                "![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=tn1994&theme=radical)")


def get_badge_as_markdown(logo: str, subject: str = None, badge_color: str = '232F3E'):
    if subject is None:
        subject = logo
    url: str = f'https://img.shields.io/badge/-{subject}-{badge_color}.svg?logo={logo}&style=flat'
    return st.markdown(f'![]({url})')
